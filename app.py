"""
競艇 AI 予測 Web アプリケーション。

起動方法:
    python app.py

ブラウザで http://localhost:5000 を開く。
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from monitor import monitor_bp

from data.scraper import scrape_race, scrape_odds, VENUE_JCD
from models.lgbm_ranking import _predict_scores, build_X
from models.kelly_betting import (
    plackett_luce_probs, proportional_allocate,
    value_bet_allocate, compute_ev_table, COMBO_STRS,
)
from models.calibration import load_calibrator, apply_calibration

app = Flask(__name__)
app.secret_key = "boatrace_ai_2026"
app.register_blueprint(monitor_bp)

MODEL_DIR = Path("models/saved")

# 会場リスト（フォーム用）
VENUES = [
    ("01", "桐生"), ("02", "戸田"), ("03", "江戸川"), ("04", "平和島"),
    ("05", "多摩川"), ("06", "浜名湖"), ("07", "蒲郡"), ("08", "常滑"),
    ("09", "津"), ("10", "三国"), ("11", "びわこ"), ("12", "住之江"),
    ("13", "尼崎"), ("14", "鳴門"), ("15", "丸亀"), ("16", "児島"),
    ("17", "宮島"), ("18", "徳山"), ("19", "下関"), ("20", "若松"),
    ("21", "芦屋"), ("22", "福岡"), ("23", "唐津"), ("24", "大村"),
]

# 枠番カラー（日本競艇公式）
LANE_COLORS = {
    1: {"bg": "#FFFFFF", "text": "#111111", "name": "白"},
    2: {"bg": "#111111", "text": "#FFFFFF", "name": "黒"},
    3: {"bg": "#D32F2F", "text": "#FFFFFF", "name": "赤"},
    4: {"bg": "#1565C0", "text": "#FFFFFF", "name": "青"},
    5: {"bg": "#F9A825", "text": "#111111", "name": "黄"},
    6: {"bg": "#2E7D32", "text": "#FFFFFF", "name": "緑"},
}


# -------------------------------------------------------
# モデルキャッシュ
# -------------------------------------------------------

_cache: dict = {}


def get_model():
    """モデルをロード（初回のみ）してキャッシュする。"""
    if "booster" not in _cache:
        if not (MODEL_DIR / "lgbm_booster.txt").exists():
            return None, None, None, None
        _cache["booster"]      = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_booster.txt"))
        _cache["scalers"]      = joblib.load(MODEL_DIR / "scalers.pkl")
        _cache["race_columns"] = joblib.load(MODEL_DIR / "race_columns.pkl")
        _cache["calibrator"]   = load_calibrator(str(MODEL_DIR / "calibrator.pkl"))
        print("[app] モデルロード完了")
    return (_cache["booster"], _cache["scalers"],
            _cache["race_columns"], _cache["calibrator"])


# -------------------------------------------------------
# 予測ロジック
# -------------------------------------------------------

def run_prediction(jcd: str, hd: str, rno: int,
                   venue_name: str, nichiji: int,
                   budget: int, use_odds: bool = True,
                   min_edge: float = 0.05,
                   kelly_frac: float = 0.25,
                   max_combos: int = 6,
                   debug: bool = False) -> dict:
    """
    スクレイピング → スケーリング → 予測 → 結果辞書を返す。
    """
    booster, scalers, race_columns, calibrator = get_model()
    if booster is None:
        raise RuntimeError("モデルが見つかりません。先に python train.py lgbm を実行してください。")

    url = (f"https://www.boatrace.jp/owpc/pc/race/racelist"
           f"?jcd={jcd}&hd={hd}&rno={rno:02d}")

    boat_features, race_features, player_names, weather, has_beforeinfo = scrape_race(
        url          = url,
        race_columns = race_columns,
        jcd          = jcd,
        hd           = hd,
        rno          = rno,
        venue_name   = venue_name,
        nichiji      = nichiji,
        debug        = debug,
    )

    # スケーリング
    boat_scaler, race_scaler = scalers
    n_boat_feats = boat_features.shape[2]
    X_boat = boat_scaler.transform(
        boat_features.reshape(-1, n_boat_feats)
    ).reshape(1, 6, n_boat_feats)
    X_race = race_scaler.transform(race_features)

    # スコア予測
    scores        = _predict_scores(booster, X_boat, X_race)[0]   # (6,)
    exp_s         = np.exp(scores - scores.max())
    boat_prob_raw = exp_s / exp_s.sum()                            # softmax (6,)

    # 確率校正（calibrator がある場合）
    if calibrator is not None:
        boat_prob = apply_calibration(calibrator, boat_prob_raw.reshape(1, 6))[0]
    else:
        boat_prob = boat_prob_raw

    rank_order = np.argsort(-scores)   # 0-indexed

    # 120 通りの PL 確率（校正済み確率から再計算）
    calibrated_scores = np.log(boat_prob + 1e-12)
    probs_120 = plackett_luce_probs(calibrated_scores)
    top10_idx = np.argsort(-probs_120)[:10]

    # オッズ取得（use_odds=True のとき）
    odds_dict: dict = {}
    if use_odds:
        try:
            odds_dict = scrape_odds(jcd, hd, rno, debug=debug)
        except Exception as e:
            print(f"[app] オッズ取得失敗: {e}")

    # 掛け金配分: オッズあり → バリューベット、なし → 比例配分
    if odds_dict:
        # 上位10予想に絞り、その中でEV > min_edge のものだけベット
        probs_top10 = np.zeros(120)
        probs_top10[top10_idx] = probs_120[top10_idx]
        bets_dict = value_bet_allocate(
            probs_top10,
            odds_dict,
            budget=budget,
            min_edge=min_edge,
            kelly_frac=kelly_frac,
            max_combos=max_combos,
        )
        bet_mode = "value"
    else:
        bets_dict = proportional_allocate(probs_120, budget=budget, min_prob_mul=2.0)
        bet_mode  = "proportional"

    total_bet   = sum(bets_dict.values())
    confidence  = float(boat_prob.max())
    conf_label  = "高" if confidence > 0.35 else "中" if confidence > 0.25 else "低"

    # 艇データ
    boats = []
    for idx in range(6):
        lane = idx + 1
        boats.append({
            "lane":      lane,
            "name":      player_names[idx] if idx < len(player_names) else f"{lane}号艇",
            "score":     round(float(scores[idx]), 3),
            "prob_pct":  round(float(boat_prob[idx]) * 100, 1),
            "pred_rank": int(np.where(rank_order == idx)[0][0]) + 1,
            **LANE_COLORS[lane],
        })

    # 3連単上位10
    trifecta = []
    for i in range(len(top10_idx)):
        idx   = top10_idx[i]
        combo = COMBO_STRS[idx]
        odds  = float(odds_dict.get(combo, 0.0))
        ev    = probs_120[idx] * odds - 1.0 if odds > 0 else None
        trifecta.append({
            "rank":     i + 1,
            "combo":    combo,
            "prob_pct": round(float(probs_120[idx]) * 100, 2),
            "odds":     round(odds, 1) if odds > 0 else None,
            "ev_pct":   round(ev * 100, 1) if ev is not None else None,
        })

    # EV テーブル（バリューベット候補の全リスト）
    ev_table = compute_ev_table(probs_120, odds_dict) if odds_dict else []

    # ベット一覧
    bets = []
    expected_return = 0.0
    expected_profit = 0.0
    for combo, amt in sorted(bets_dict.items(), key=lambda x: -x[1]):
        combo_idx = COMBO_STRS.index(combo)
        odds = float(odds_dict.get(combo, 0.0))
        prob = float(probs_120[combo_idx])
        ev   = prob * odds - 1.0 if odds > 0 else None
        if ev is not None:
            expected_return += amt * prob * odds
            expected_profit += amt * ev
        bets.append({
            "combo":    combo,
            "amount":   amt,
            "bar_pct":  round(amt / budget * 100),
            "prob_pct": round(prob * 100, 2),
            "odds":     round(odds, 1) if odds > 0 else None,
            "ev_pct":   round(ev * 100, 1) if ev is not None else None,
        })

    primary_combo = bets[0]["combo"] if bets else (trifecta[0]["combo"] if trifecta else "—")
    primary_idx = COMBO_STRS.index(primary_combo) if primary_combo in COMBO_STRS else None
    primary_odds = float(odds_dict.get(primary_combo, 0.0)) if primary_idx is not None else 0.0
    primary_prob = float(probs_120[primary_idx]) if primary_idx is not None else 0.0
    primary_ev = primary_prob * primary_odds - 1.0 if primary_odds > 0 else None

    # 日付フォーマット
    date_fmt = f"{hd[:4]}年{int(hd[4:6])}月{int(hd[6:8])}日" if len(hd) == 8 else hd

    return {
        "race": {
            "venue":       venue_name,
            "date":        date_fmt,
            "race_num":    rno,
            "distance":    weather.get("距離", 1800),
            "weather":     weather.get("天候", "—"),
            "wind_dir":    weather.get("風向", "—"),
            "wind_speed":  weather.get("風速", 0.0),
            "wave_height": weather.get("波高", 0.0),
        },
        "boats":       sorted(boats, key=lambda b: b["pred_rank"]),
        "boats_by_lane": sorted(boats, key=lambda b: b["lane"]),
        "trifecta":    trifecta,
        "bets":        bets,
        "budget":      budget,
        "total_bet":   total_bet,
        "unused_budget": budget - total_bet,
        "budget_used_pct": round(total_bet / budget * 100) if budget > 0 else 0,
        "expected_return": round(expected_return),
        "expected_profit": round(expected_profit),
        "expected_roi_pct": round(expected_profit / total_bet * 100, 1) if total_bet > 0 else None,
        "confidence":  round(confidence * 100, 1),
        "conf_label":  conf_label,
        "top_combo":      trifecta[0]["combo"] if trifecta else "—",
        "primary_combo":  primary_combo,
        "primary_prob_pct": round(primary_prob * 100, 2),
        "primary_odds":  round(primary_odds, 1) if primary_odds > 0 else None,
        "primary_ev_pct": round(primary_ev * 100, 1) if primary_ev is not None else None,
        "has_beforeinfo": has_beforeinfo,
        "has_odds":       bool(odds_dict),
        "bet_mode":       bet_mode,
        "min_edge_pct":   round(min_edge * 100, 1),
        "kelly_frac_pct": round(kelly_frac * 100, 1),
        "max_combos":     max_combos,
        "ev_table":       ev_table[:20],   # EV上位20通りをテンプレートに渡す
        # orchestrator が複数戦略を走らせるために使う内部データ
        "_probs_120":     probs_120,
        "_odds_dict":     odds_dict,
    }


# -------------------------------------------------------
# ルート
# -------------------------------------------------------

@app.route("/")
def index():
    booster, _, _, _ = get_model()
    model_ready = booster is not None
    return render_template("index.html",
                           venues=VENUES,
                           today=date.today().strftime("%Y-%m-%d"),
                           model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    mode   = request.form.get("mode", "url")
    budget = int(request.form.get("budget", 1000))
    debug  = request.form.get("debug") == "1"

    try:
        if mode == "url":
            url = request.form.get("url", "").strip()
            if not url:
                flash("URL を入力してください。", "danger")
                return redirect(url_for("index"))

            from data.scraper import _url_to_jcd_hd_rno
            jcd, hd, rno = _url_to_jcd_hd_rno(url)

        else:
            jcd = request.form.get("venue", "04")
            raw_date = request.form.get("date", "").replace("-", "")
            hd  = raw_date if len(raw_date) == 8 else date.today().strftime("%Y%m%d")
            rno = int(request.form.get("race", 1))

        venue_name = dict(VENUES).get(jcd, "—")
        nichiji    = int(request.form.get("nichiji", 1))
        use_odds   = request.form.get("use_odds", "1") != "0"
        min_edge   = float(request.form.get("min_edge", "0.05"))
        kelly_frac = float(request.form.get("kelly_frac", "0.25"))
        max_combos = int(request.form.get("max_combos", "6"))

        result = run_prediction(
            jcd, hd, rno, venue_name, nichiji, budget,
            use_odds=use_odds,
            min_edge=min_edge,
            kelly_frac=kelly_frac,
            max_combos=max_combos,
            debug=debug,
        )
        return render_template("result.html", r=result)

    except Exception as exc:
        flash(f"エラー: {exc}", "danger")
        return redirect(url_for("index"))


if __name__ == "__main__":
    print("=" * 50)
    print("  競艇 AI 予測サーバー起動")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)

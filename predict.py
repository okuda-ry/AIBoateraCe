"""
リアルタイム予測スクリプト。

使い方:
    # 平和島 1R（今日）
    python predict.py --url "https://www.heiwajima.gr.jp/asp/heiwajima/kyogi/kyogihtml/index.htm?racenum=1"

    # 公式サイト直接指定
    python predict.py --url "https://www.boatrace.jp/owpc/pc/race/racelist?jcd=04&hd=20260320&rno=1"

    # 会場・日付・レース番号で指定
    python predict.py --venue 04 --date 20260320 --race 1

    # 予算変更（デフォルト 1000 円）
    python predict.py --url "..." --budget 2000

    # デバッグモード（スクレイピング詳細を表示）
    python predict.py --url "..." --debug
"""

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from data.scraper import scrape_race, scrape_odds, VENUE_JCD
from models.lgbm_ranking import build_X, _predict_scores
from models.kelly_betting import (
    plackett_luce_probs, proportional_allocate,
    value_bet_allocate, compute_ev_table, COMBO_STRS
)
from models.calibration import load_calibrator, apply_calibration

MODEL_DIR = Path("models/saved")


# -------------------------------------------------------
# モデルロード
# -------------------------------------------------------

def load_model() -> tuple:
    """保存済みモデルとメタデータをロードする。"""
    if not (MODEL_DIR / "lgbm_booster.txt").exists():
        print("[ERROR] モデルが見つかりません。先に `python train.py lgbm` を実行してください。")
        sys.exit(1)

    booster      = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_booster.txt"))
    scalers      = joblib.load(MODEL_DIR / "scalers.pkl")
    race_columns = joblib.load(MODEL_DIR / "race_columns.pkl")
    calibrator   = load_calibrator(str(MODEL_DIR / "calibrator.pkl"))

    return booster, scalers, race_columns, calibrator


# -------------------------------------------------------
# 特徴量スケーリング
# -------------------------------------------------------

def scale_features(boat_features: np.ndarray,
                   race_features: np.ndarray,
                   scalers: tuple) -> tuple:
    """
    学習時と同じスケーリングを適用する。

    Parameters
    ----------
    boat_features : (1, 6, 23)
    race_features : (1, n_race)
    scalers       : (boat_scaler, race_scaler)
    """
    boat_scaler, race_scaler = scalers
    n_boat_feats = boat_features.shape[2]

    X_boat = boat_scaler.transform(
        boat_features.reshape(-1, n_boat_feats)
    ).reshape(1, 6, n_boat_feats)

    X_race = race_scaler.transform(race_features)

    return X_boat, X_race


# -------------------------------------------------------
# 予測・表示
# -------------------------------------------------------

def predict_and_display(booster: lgb.Booster,
                        X_boat: np.ndarray,
                        X_race: np.ndarray,
                        budget: int = 1000,
                        player_names: list[str] = None,
                        odds_dict: dict = None,
                        calibrator=None):
    """
    スコアを計算し、予測結果と推奨ベットを表示する。

    Parameters
    ----------
    odds_dict : scrape_odds() で取得したオッズ辞書。
                指定時はバリューベット（EV>0の組のみ）を行う。
                None の場合は比例配分にフォールバックする。
    """
    scores = _predict_scores(booster, X_boat, X_race)   # (1, 6)
    scores_1 = scores[0]

    # softmax 確率（生）
    exp_s      = np.exp(scores_1 - scores_1.max())
    boat_probs_raw = exp_s / exp_s.sum()

    # 校正確率（calibrator がある場合）
    if calibrator is not None:
        boat_probs = apply_calibration(calibrator, boat_probs_raw.reshape(1, 6))[0]
    else:
        boat_probs = boat_probs_raw

    confidence = boat_probs.max()

    # ランキング（生スコアで順序付け）
    rank_order = np.argsort(-scores_1)   # 0-indexed, descending

    # 120 通りの PL 確率（校正済み確率から再計算）
    # 校正済みの確率をスコアとして扱い、PL確率を計算する
    # ログ空間に戻してPlackett-Luceを再計算
    calibrated_scores = np.log(boat_probs + 1e-12)
    probs_120 = plackett_luce_probs(calibrated_scores)
    top_idx   = np.argsort(-probs_120)[:10]

    # ベット配分: オッズあり → バリューベット、なし → 比例配分
    if odds_dict:
        bets        = value_bet_allocate(probs_120, odds_dict, budget=budget, min_edge=0.05)
        bet_mode    = "バリューベット（期待値>5%）"
        if not bets:
            bets     = {}
            bet_mode = "バリューベット — 期待値プラスの組なし → 見送り推奨"
    else:
        bets     = proportional_allocate(probs_120, budget=budget, min_prob_mul=2.0)
        bet_mode = "比例配分（オッズ未取得）"

    # -------------------------------------------------------
    print()
    print("=" * 55)
    print("  3連単 予測結果")
    print("=" * 55)

    if player_names is None:
        player_names = [f"{i}号艇" for i in range(1, 7)]

    print("\n【艇別スコア】")
    print(f"  {'枠':>3}  {'選手':>12}  {'スコア':>8}  {'勝率予測':>8}  {'順位予測'}")
    print("  " + "-" * 50)
    for r, idx in enumerate(rank_order, 1):
        print(f"  {idx+1:>3}  {player_names[idx]:>12}  "
              f"{scores_1[idx]:>8.3f}  {boat_probs[idx]*100:>7.1f}%  "
              f"{'★' * max(0, 4-r)}")

    print(f"\n  信頼度 (1着予測艇 softmax): {confidence*100:.1f}%  "
          f"({'高' if confidence > 0.35 else '中' if confidence > 0.25 else '低'})")

    print("\n【3連単 上位10通り】")
    if odds_dict:
        print(f"  {'順位':>4}  {'組番':>8}  {'確率':>8}  {'オッズ':>8}  {'期待値':>8}")
        print("  " + "-" * 48)
        for rank, i in enumerate(top_idx, 1):
            combo = COMBO_STRS[i]
            odds  = odds_dict.get(combo, 0.0)
            ev    = probs_120[i] * odds - 1.0 if odds > 0 else float("nan")
            ev_str = f"{ev*100:>+7.1f}%" if odds > 0 else "    N/A"
            marker = " ◆" if ev > 0.05 else ""
            print(f"  {rank:>4}  {combo:>8}  {probs_120[i]*100:>7.2f}%  "
                  f"{odds:>7.1f}x  {ev_str}{marker}")
    else:
        print(f"  {'順位':>4}  {'組番':>8}  {'確率':>8}")
        print("  " + "-" * 28)
        for rank, i in enumerate(top_idx, 1):
            print(f"  {rank:>4}  {COMBO_STRS[i]:>8}  {probs_120[i]*100:>7.2f}%")

    print(f"\n【推奨ベット（{budget}円配分・{bet_mode}）】")
    if bets:
        total = sum(bets.values())
        if odds_dict:
            print(f"  {'組番':>8}  {'金額':>8}  {'オッズ':>8}  {'期待値':>8}")
            print("  " + "-" * 38)
            for combo, yen in sorted(bets.items(), key=lambda x: -x[1]):
                odds = odds_dict.get(combo, 0.0)
                ev   = probs_120[COMBO_STRS.index(combo)] * odds - 1.0 if odds > 0 else 0
                bar  = "█" * (yen // 100)
                print(f"  {combo:>8}  {yen:>6}円  {odds:>7.1f}x  {ev*100:>+6.1f}%  {bar}")
        else:
            print(f"  {'組番':>8}  {'金額':>8}")
            print("  " + "-" * 20)
            for combo, yen in sorted(bets.items(), key=lambda x: -x[1]):
                bar = "█" * (yen // 100)
                print(f"  {combo:>8}  {yen:>6}円  {bar}")
        print(f"  {'合計':>8}  {total:>6}円")
    else:
        print("  → 見送りを推奨（条件を満たす組み合わせなし）")

    print()
    return scores_1, probs_120, bets


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="競艇 AI リアルタイム予測")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--url",   help="出走表 URL")
    parser.add_argument("--venue", default=None,
                        help="会場コード (例: 04=平和島)")
    parser.add_argument("--date",  default=None,
                        help="日付 YYYYMMDD (例: 20260320)")
    parser.add_argument("--race",  type=int, default=1,
                        help="レース番号 (default: 1)")
    parser.add_argument("--venue-name", default="平和島",
                        help="レース場名（one-hot 用）")
    parser.add_argument("--nichiji", type=int, default=1,
                        help="日次（何日目か）")
    parser.add_argument("--budget", type=int, default=1000,
                        help="1レース予算 (円, default: 1000)")
    parser.add_argument("--no-odds", action="store_true",
                        help="オッズ取得をスキップして比例配分を使う")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="バリューベットの最低期待値 (default: 0.05 = 5%%)")
    parser.add_argument("--debug", action="store_true",
                        help="スクレイピング詳細を表示")
    args = parser.parse_args()

    # URL または venue/date/race からパラメータを決定
    if args.url:
        url  = args.url
        jcd  = None
        hd   = None
        rno  = None
    elif args.venue:
        from datetime import date as _date
        jcd  = VENUE_JCD.get(args.venue, args.venue)
        hd   = args.date or _date.today().strftime("%Y%m%d")
        rno  = args.race
        url  = (f"https://www.boatrace.jp/owpc/pc/race/racelist"
                f"?jcd={jcd}&hd={hd}&rno={rno:02d}")
    else:
        parser.print_help()
        sys.exit(1)

    # モデルロード
    booster, scalers, race_columns, calibrator = load_model()

    # jcd/hd/rno を確定（URL からでも引数からでも）
    if args.url and (jcd is None or hd is None or rno is None):
        from data.scraper import _url_to_jcd_hd_rno
        jcd, hd, rno = _url_to_jcd_hd_rno(url)

    # スクレイピング（出走表 + 直前情報）
    boat_features, race_features, player_names, weather, has_beforeinfo = scrape_race(
        url           = url,
        race_columns  = race_columns,
        jcd           = jcd,
        hd            = hd,
        rno           = rno,
        venue_name    = args.venue_name,
        nichiji       = args.nichiji,
        debug         = args.debug,
    )

    # オッズ取得（--no-odds なら省略）
    odds_dict = {}
    if not args.no_odds:
        print(f"[predict] 3連単オッズ取得中 (jcd={jcd} hd={hd} rno={rno:02d})")
        odds_dict = scrape_odds(jcd, hd, rno, debug=args.debug)

    # スケーリング
    X_boat, X_race = scale_features(boat_features, race_features, scalers)

    # 予測・表示
    predict_and_display(booster, X_boat, X_race,
                        budget=args.budget, player_names=player_names,
                        odds_dict=odds_dict, calibrator=calibrator)


if __name__ == "__main__":
    main()

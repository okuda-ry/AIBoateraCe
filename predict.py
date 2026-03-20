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

from data.scraper import scrape_race, VENUE_JCD
from models.lgbm_ranking import build_X, _predict_scores
from models.kelly_betting import (
    plackett_luce_probs, proportional_allocate, COMBO_STRS
)

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

    return booster, scalers, race_columns


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
                        player_names: list[str] = None):
    """
    スコアを計算し、予測結果と推奨ベットを表示する。
    """
    scores = _predict_scores(booster, X_boat, X_race)   # (1, 6)
    scores_1 = scores[0]

    # softmax 確率
    exp_s      = np.exp(scores_1 - scores_1.max())
    boat_probs = exp_s / exp_s.sum()
    confidence = boat_probs.max()

    # ランキング
    rank_order = np.argsort(-scores_1)   # 0-indexed, descending

    # 120 通りの PL 確率
    probs_120 = plackett_luce_probs(scores_1)
    top_idx   = np.argsort(-probs_120)[:10]

    # Kelly 比例配分
    bets = proportional_allocate(probs_120, budget=budget, min_prob_mul=2.0)

    # -------------------------------------------------------
    print()
    print("=" * 50)
    print("  3連単 予測結果")
    print("=" * 50)

    if player_names is None:
        player_names = [f"{i}号艇" for i in range(1, 7)]

    print("\n【艇別スコア】")
    print(f"  {'枠':>3}  {'選手':>12}  {'スコア':>8}  {'勝率予測':>8}  {'順位予測'}")
    print("  " + "-" * 48)
    for r, idx in enumerate(rank_order, 1):
        print(f"  {idx+1:>3}  {player_names[idx]:>12}  "
              f"{scores_1[idx]:>8.3f}  {boat_probs[idx]*100:>7.1f}%  "
              f"{'★' * max(0, 4-r)}")

    print(f"\n  信頼度 (1着予測艇 softmax): {confidence*100:.1f}%  "
          f"({'高' if confidence > 0.35 else '中' if confidence > 0.25 else '低'})")

    print("\n【3連単 上位10通り】")
    print(f"  {'順位':>4}  {'組番':>8}  {'確率':>8}")
    print("  " + "-" * 28)
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:>4}  {COMBO_STRS[i]:>8}  {probs_120[i]*100:>7.2f}%")

    print(f"\n【推奨ベット（{budget}円配分・比例配分）】")
    if bets:
        total = sum(bets.values())
        print(f"  {'組番':>8}  {'金額':>8}")
        print("  " + "-" * 20)
        for combo, yen in sorted(bets.items(), key=lambda x: -x[1]):
            bar = "█" * (yen // 100)
            print(f"  {combo:>8}  {yen:>6}円  {bar}")
        print(f"  {'合計':>8}  {total:>6}円")
    else:
        print("  → 確信度が低いため見送りを推奨")

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
    booster, scalers, race_columns = load_model()

    # スクレイピング
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

    # スケーリング
    X_boat, X_race = scale_features(boat_features, race_features, scalers)

    # 予測・表示
    predict_and_display(booster, X_boat, X_race,
                        budget=args.budget, player_names=player_names)


if __name__ == "__main__":
    main()

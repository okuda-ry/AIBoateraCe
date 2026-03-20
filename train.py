"""
学習エントリーポイント。

使い方:
    python train.py lgbm      # LightGBM LambdaRank（推奨・デフォルト）
    python train.py ranking   # Self-Attention ランキングネット
    python train.py baseline  # 単勝ベースライン
"""

import argparse
import sys

from data.preprocess import load_and_merge, build_dataset, split_and_scale
from models import ranking, baseline, lgbm_ranking
from models.kelly_betting import compare_strategies


def _get_race_columns(df) -> list:
    """build_race_features と同じ列名リストを返す。"""
    import pandas as pd
    num_cols = ["風速", "波高", "日次", "距離"]
    ohe_df   = pd.get_dummies(df[["天候", "風向", "レース場"]],
                               prefix=["天候", "風向", "場"])
    return num_cols + list(ohe_df.columns)


def _save_model(booster, scalers, race_cols: list):
    """学習済みモデル・スケーラー・特徴量メタデータを保存する。"""
    import os, joblib

    save_dir = "models/saved"
    os.makedirs(save_dir, exist_ok=True)

    booster.save_model(f"{save_dir}/lgbm_booster.txt")
    joblib.dump(scalers,    f"{save_dir}/scalers.pkl")
    joblib.dump(race_cols,  f"{save_dir}/race_columns.pkl")

    print(f"[save] {save_dir}/ に保存: lgbm_booster.txt, scalers.pkl, race_columns.pkl")


def train_lgbm(timetable_path: str, details_path: str):
    """LightGBM LambdaRank を学習・評価する（推奨）。"""
    df = load_and_merge(timetable_path, details_path)

    boat_features, race_features, positions, valid_mask, trifecta_str, trifecta_pay = (
        build_dataset(df)
    )

    train_data, val_data, test_data, scalers = split_and_scale(
        boat_features, race_features, positions,
        valid_mask, trifecta_str, trifecta_pay,
    )

    booster = lgbm_ranking.train(train_data, val_data)
    lgbm_ranking.evaluate(booster, test_data)
    lgbm_ranking.evaluate_threshold(booster, test_data)

    # 掛け金最適化バックテスト
    X_boat_te, X_race_te, _, y_tri_str_te, y_tri_pay_te = test_data
    scores_all = lgbm_ranking._predict_scores(booster, X_boat_te, X_race_te)
    compare_strategies(scores_all, y_tri_str_te, y_tri_pay_te, budget=1000)

    # モデル保存
    race_cols = _get_race_columns(df)
    _save_model(booster, scalers, race_cols)


def train_ranking(timetable_path: str, details_path: str):
    """Self-Attention ランキングネットを学習・評価する（ペイアウト重み付き）。"""
    df = load_and_merge(timetable_path, details_path)

    boat_features, race_features, positions, valid_mask, trifecta_str, trifecta_pay = (
        build_dataset(df)
    )

    train_data, val_data, test_data, _ = split_and_scale(
        boat_features, race_features, positions,
        valid_mask, trifecta_str, trifecta_pay,
    )

    n_boat_feats = train_data[0].shape[2]
    n_race_feats = train_data[1].shape[1]

    model = ranking.build_attention_model(n_boat_feats, n_race_feats)
    model.summary()

    ranking.train(model, train_data, val_data)
    top1_hits, top1_gains, top1_rr, top4_hits, top4_gains, top4_rr = (
        ranking.evaluate(model, test_data)
    )

    # 信頼度閾値分析
    ranking.evaluate_threshold(model, test_data)


def train_baseline(timetable_path: str, details_path: str):
    """単勝ベースライン分類モデルを学習・評価する。"""
    import numpy as np
    import pandas as pd
    import unicodedata, re
    from sklearn.preprocessing import StandardScaler
    from data.preprocess import normalize_date, normalize_race_round

    # --- データ読み込み ---
    df = load_and_merge(timetable_path, details_path)

    # --- 特徴量（全枠を横に並べたフラットベクトル）---
    key_patterns = ["日次", "レース場", "年齢", "体重", "級別",
                    "全国勝率", "全国2連対率", "当地勝率", "当地2連対率",
                    "モーター2連対率", "ボート2連対率", "早見",
                    "天候", "風向", "風速", "波の高さ"]
    cols = [c for c in df.columns if any(k in c for k in key_patterns)]
    cols = [c for c in cols if c not in ("日次_x", "日次_y")]

    feat_df = df[cols].copy()
    feat_df = pd.get_dummies(feat_df, columns=[c for c in feat_df.columns
                                               if feat_df[c].dtype == object])
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values).astype(np.float32)

    y_boat = pd.to_numeric(df["単勝_艇番"],  errors="coerce")
    y_pay  = pd.to_numeric(df["単勝_払戻金"], errors="coerce")
    valid  = y_boat.notna()

    X = X[valid]; y_boat = y_boat[valid].values.astype(int); y_pay = y_pay[valid].values

    n = len(X)
    i1, i2 = int(n * 0.70), int(n * 0.80)
    X_tr, X_va, X_te = X[:i1], X[i1:i2], X[i2:]
    y_tr, y_va, y_te = y_boat[:i1], y_boat[i1:i2], y_boat[i2:]
    p_te = y_pay[i2:]

    model = baseline.build_model(X.shape[1])
    model.summary()

    baseline.train(model, (X_tr, y_tr), (X_va, y_va))
    baseline.evaluate(model, (X_te, y_te, p_te))


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競艇 AI 学習スクリプト")
    parser.add_argument(
        "model", nargs="?", default="lgbm",
        choices=["lgbm", "ranking", "baseline"],
        help="学習するモデル (default: lgbm)",
    )
    parser.add_argument(
        "--timetable",
        default="downloads/racelists/csv/timetable_200901-240901.csv",
        help="出走表 CSV パス",
    )
    parser.add_argument(
        "--details",
        default="downloads/results/details/details_200901-240901.csv",
        help="競走成績詳細 CSV パス",
    )
    args = parser.parse_args()

    if args.model == "lgbm":
        train_lgbm(args.timetable, args.details)
    elif args.model == "ranking":
        train_ranking(args.timetable, args.details)
    else:
        train_baseline(args.timetable, args.details)

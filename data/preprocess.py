"""
CSV の読み込み・マージ・特徴量エンジニアリング。
モデル学習に使うテンソルを生成する。
"""

import unicodedata
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# デフォルトデータパス
# -------------------------------------------------------
TIMETABLE_PATH = "downloads/racelists/csv/timetable_200901-240901.csv"
DETAILS_PATH   = "downloads/results/details/details_200901-240901.csv"

# -------------------------------------------------------
# 艇ごとの特徴量カラム名
# -------------------------------------------------------
BOAT_NUM_FEATS   = [
    "年齢", "体重", "全国勝率", "全国2連対率",
    "当地勝率", "当地2連対率", "モーター2連対率", "ボート2連対率", "早見",
    "今節平均ST",  # 今節の平均スタートタイム (小さいほど早い = 有利)
]
BOAT_TODAY_FEATS = [f"今節成績_{i}-{j}" for i in range(1, 7) for j in range(1, 3)]
KYUBETSU_MAP     = {"A1": 3.0, "A2": 2.0, "B1": 1.0, "B2": 0.0}


# -------------------------------------------------------
# 日付・レース回の正規化
# -------------------------------------------------------

def normalize_date(s: str) -> str:
    """'YYYY年MM月DD日' → 'YYYY/MM/DD'"""
    s = unicodedata.normalize("NFKC", str(s)).replace(" ", "")
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}/{int(mo):02}/{int(d):02}"
    return s


def normalize_race_round(s: str) -> str:
    """'1R' / '１Ｒ' → '01R'"""
    s = unicodedata.normalize("NFKC", str(s)).replace(" ", "")
    m = re.match(r"(\d+)R", s.upper())
    if m:
        return f"{int(m.group(1)):02}R"
    return s


# -------------------------------------------------------
# データ読み込みとマージ
# -------------------------------------------------------

def load_and_merge(timetable_path: str = TIMETABLE_PATH,
                   details_path:   str = DETAILS_PATH) -> pd.DataFrame:
    """
    出走表 CSV と競走成績詳細 CSV を読み込んでマージする。
    """
    timetable_df = pd.read_csv(timetable_path, encoding="shift-jis", low_memory=False)
    details_df   = pd.read_csv(details_path,   encoding="shift-jis", low_memory=False)

    timetable_df["レース日"] = timetable_df["レース日"].apply(normalize_date)
    timetable_df["レース回"] = timetable_df["レース回"].apply(normalize_race_round)

    df = timetable_df.merge(details_df, on=["レース日", "レース場", "レース回"])
    df["日次"] = df["日次_x"].str.extract(r"(\d+)").astype(int)

    print(f"マージ後レース数: {len(df):,}")
    return df


# -------------------------------------------------------
# ターゲット（着順）
# -------------------------------------------------------

def build_positions(df: pd.DataFrame) -> np.ndarray:
    """
    各枠（1〜6）の着順を返す。shape: (n_races, 6)

    例: 1着_艇番=3 → positions[race, 2] = 1.0
    """
    n = len(df)
    positions = np.zeros((n, 6), dtype=np.float32)
    for rank in range(1, 7):
        boat_nums = pd.to_numeric(df[f"{rank}着_艇番"], errors="coerce").values
        for boat in range(1, 7):
            positions[boat_nums == boat, boat - 1] = float(rank)
    return positions


# -------------------------------------------------------
# 特徴量
# -------------------------------------------------------

def build_boat_features(df: pd.DataFrame) -> np.ndarray:
    """
    艇ごとの特徴量。shape: (n_races, 6, n_boat_feats)

    各艇の特徴量:
        年齢・体重・全国勝率・全国2連対率・当地勝率・当地2連対率・
        モーター2連対率・ボート2連対率・早見 (9 列)
        今節成績_1-1 〜 今節成績_6-2               (12 列)
        級別エンコード (A1=3, A2=2, B1=1, B2=0)    ( 1 列)
        枠番 (1〜6)                                  ( 1 列)
        計 23 次元
    """
    feat_list = []
    for n in range(1, 7):
        p     = f"{n}枠_"
        # 今節平均ST は CSV に存在しない場合があるので個別に処理
        base_feats = [f for f in BOAT_NUM_FEATS if f != "今節平均ST"]
        num_base = df[[p + f for f in base_feats]].apply(pd.to_numeric, errors="coerce").fillna(0).values

        st_col = p + "今節平均ST"
        if st_col in df.columns:
            st_vals = pd.to_numeric(df[st_col], errors="coerce").fillna(0.18).values.reshape(-1, 1)
        else:
            # CSVに列がない場合は全国平均 (0.18) で埋める
            st_vals = np.full((len(df), 1), 0.18, dtype=np.float32)

        num   = np.concatenate([num_base, st_vals], axis=1)
        today = df[[p + f for f in BOAT_TODAY_FEATS]].apply(pd.to_numeric, errors="coerce").fillna(0).values
        kyu   = df[p + "級別"].map(KYUBETSU_MAP).fillna(0).values.reshape(-1, 1)
        lane  = np.full((len(df), 1), float(n), dtype=np.float32)  # 枠番 1-6
        feat_list.append(np.concatenate([num, today, kyu, lane], axis=1))
    return np.stack(feat_list, axis=1).astype(np.float32)


def build_race_features(df: pd.DataFrame) -> np.ndarray:
    """
    レースレベルの特徴量。shape: (n_races, n_race_feats)

    内訳: 風速・波高・日次・距離（数値）+ 天候・風向・レース場（one-hot）
    """
    num_df = pd.DataFrame({
        "風速": pd.to_numeric(df["風速(m)"],       errors="coerce").fillna(0),
        "波高": pd.to_numeric(df["波の高さ(cm)"],  errors="coerce").fillna(0),
        "日次": df["日次"].values,
        "距離": pd.to_numeric(df["距離(m)_x"],     errors="coerce").fillna(1800),
    })
    ohe_df = pd.get_dummies(df[["天候", "風向", "レース場"]], prefix=["天候", "風向", "場"])
    return pd.concat([num_df, ohe_df], axis=1).fillna(0).astype(np.float32).values


# -------------------------------------------------------
# データセット構築
# -------------------------------------------------------

def build_dataset(df: pd.DataFrame):
    """
    全特徴量・ターゲット・有効マスクをまとめて返す。

    Returns
    -------
    boat_features : (n_races, 6, n_boat_feats)
    race_features : (n_races, n_race_feats)
    positions     : (n_races, 6)  — 各枠の着順
    valid_mask    : (n_races,)    — 全6艇の着順が揃っているレース
    trifecta_str  : (n_races,)    — 3連単組番 "a-b-c"
    trifecta_pay  : (n_races,)    — 3連単払戻金
    """
    boat_features = build_boat_features(df)
    race_features = build_race_features(df)
    positions     = build_positions(df)
    valid_mask    = (positions > 0).all(axis=1)
    trifecta_str  = df["3連単_組番"].astype(str).str.strip().values
    trifecta_pay  = pd.to_numeric(df["3連単_払戻金"], errors="coerce").fillna(0).values

    print(f"有効レース: {valid_mask.sum():,} / {len(df):,} ({valid_mask.mean()*100:.1f}%)")
    return boat_features, race_features, positions, valid_mask, trifecta_str, trifecta_pay


# -------------------------------------------------------
# 分割・正規化
# -------------------------------------------------------

def split_and_scale(boat_features, race_features, positions,
                    valid_mask, trifecta_str, trifecta_pay,
                    train_ratio: float = 0.70,
                    val_ratio:   float = 0.10):
    """
    有効レースのみ抽出し、時系列順に train / val / test へ分割して正規化する。

    正規化は train データのみで fit し、val・test に適用（データリーク防止）。

    Returns
    -------
    train_data : (X_boat_tr, X_race_tr, y_tr, pay_tr)
    val_data   : (X_boat_va, X_race_va, y_va, pay_va)
    test_data  : (X_boat_te, X_race_te, y_te, trifecta_str_te, trifecta_pay_te)
    scalers    : (boat_scaler, race_scaler)
    """
    X_boat = boat_features[valid_mask]
    X_race = race_features[valid_mask]
    y_pos  = positions[valid_mask]
    y_tstr = trifecta_str[valid_mask]
    y_tpay = trifecta_pay[valid_mask]

    n  = len(X_boat)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    n_boat_feats = X_boat.shape[2]

    boat_scaler = StandardScaler()
    X_boat_tr = boat_scaler.fit_transform(X_boat[:i1].reshape(-1, n_boat_feats)).reshape(-1, 6, n_boat_feats)
    X_boat_va = boat_scaler.transform(X_boat[i1:i2].reshape(-1, n_boat_feats)).reshape(-1, 6, n_boat_feats)
    X_boat_te = boat_scaler.transform(X_boat[i2:].reshape(-1, n_boat_feats)).reshape(-1, 6, n_boat_feats)

    race_scaler = StandardScaler()
    X_race_tr = race_scaler.fit_transform(X_race[:i1])
    X_race_va = race_scaler.transform(X_race[i1:i2])
    X_race_te = race_scaler.transform(X_race[i2:])

    print(f"Train: {i1:,}  Val: {i2 - i1:,}  Test: {n - i2:,}")
    print(f"艇特徴量: {X_boat_tr.shape}  レース特徴量: {X_race_tr.shape}")

    train_data = (X_boat_tr, X_race_tr, y_pos[:i1],      y_tpay[:i1])
    val_data   = (X_boat_va, X_race_va, y_pos[i1:i2],    y_tpay[i1:i2])
    test_data  = (X_boat_te, X_race_te, y_pos[i2:], y_tstr[i2:], y_tpay[i2:])
    scalers    = (boat_scaler, race_scaler)

    return train_data, val_data, test_data, scalers

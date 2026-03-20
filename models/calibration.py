"""
確率校正モジュール — Isotonic Regression による softmax 補正。

LightGBM の softmax (Plackett-Luce) 出力は「真の勝率」に対して
バイアスを持つことがあります。特に:
  - 高確率帯を過大評価 / 低確率帯を過小評価（または逆）
  - ランキング損失で学習しているため確率の絶対値は副産物

Isotonic Regression で「生のsoftmax確率 → 真の当選率」へ
単調に変換することで、EV計算・Kelly配分の精度が向上します。

使い方 (train.py 内):
    calibrator = fit_calibrator(val_probs, val_positions)
    save_calibrator(calibrator, "models/saved/calibrator.pkl")

使い方 (predict.py 内):
    calibrator = load_calibrator("models/saved/calibrator.pkl")
    cal_probs = apply_calibration(calibrator, raw_boat_probs)
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression


# -------------------------------------------------------
# 学習
# -------------------------------------------------------

def fit_calibrator(boat_probs: np.ndarray,
                   positions: np.ndarray) -> IsotonicRegression:
    """
    検証データで Isotonic Regression 校正器を学習する。

    Parameters
    ----------
    boat_probs : (n_races, 6)  LightGBM の softmax 確率（未校正）
    positions  : (n_races, 6)  着順（1=1着, 2=2着, ...）

    Returns
    -------
    calibrator : 学習済み IsotonicRegression
    """
    # 1着バイナリラベル (1=勝利, 0=それ以外)
    win_labels = (positions == 1).astype(float).ravel()   # (n*6,)
    raw_probs  = boat_probs.ravel()                        # (n*6,)

    # Isotonic Regression: 単調増加制約で真の当選率に近似
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(raw_probs, win_labels)

    # 校正前後の診断
    cal_probs = ir.predict(raw_probs)
    print("=== 確率校正 (Isotonic Regression) ===")
    print(f"  サンプル数 (6艇 × レース数): {len(raw_probs):,}")
    print(f"  生 softmax — mean: {raw_probs.mean():.4f}  "
          f"1着艇の平均: {raw_probs[win_labels == 1].mean():.4f}")
    print(f"  校正後     — mean: {cal_probs.mean():.4f}  "
          f"1着艇の平均: {cal_probs[win_labels == 1].mean():.4f}")
    print(f"  理想値 (1/6 = 0.1667): 1着艇の平均が 1.0 に近いほど良い")

    # Brier スコアで校正精度を確認
    brier_before = float(np.mean((raw_probs - win_labels) ** 2))
    brier_after  = float(np.mean((cal_probs  - win_labels) ** 2))
    print(f"  Brier スコア: 校正前={brier_before:.6f}  校正後={brier_after:.6f}")

    return ir


# -------------------------------------------------------
# 適用
# -------------------------------------------------------

def apply_calibration(calibrator: IsotonicRegression,
                      boat_probs: np.ndarray) -> np.ndarray:
    """
    校正器を適用し、確率を正規化して返す。

    Parameters
    ----------
    calibrator : 学習済み IsotonicRegression（None の場合は無変換）
    boat_probs : (6,) または (n_races, 6)  softmax 確率

    Returns
    -------
    calibrated : 同形状、行ごとに合計=1 に正規化
    """
    if calibrator is None:
        return boat_probs

    shape = boat_probs.shape
    flat  = boat_probs.ravel()
    cal   = calibrator.predict(flat).reshape(shape).astype(np.float32)

    # 行ごとに正規化（合計=1）
    if cal.ndim == 2:
        row_sum = cal.sum(axis=1, keepdims=True) + 1e-12
        cal     = cal / row_sum
    else:
        cal = cal / (cal.sum() + 1e-12)

    return cal


# -------------------------------------------------------
# 保存 / 読み込み
# -------------------------------------------------------

def save_calibrator(calibrator: IsotonicRegression, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)
    print(f"[calibration] 校正器を保存: {path}")


def load_calibrator(path: str):
    """
    保存済み校正器をロードする。ファイルがなければ None を返す。
    """
    p = Path(path)
    if not p.exists():
        return None
    cal = joblib.load(p)
    print(f"[calibration] 校正器をロード: {path}")
    return cal

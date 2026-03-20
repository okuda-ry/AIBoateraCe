"""
LightGBM LambdaRank — 3連単予測モデル。

LightGBM の lambdarank objective で各艇のスコアを学習し、
上位3艇の順序から3連単を予測する。

DNNより表形式データに強く、特徴量重要度の解釈も容易。

特徴量（per boat per race）:
    - 艇特徴量: 23次元（preprocess.py で定義）
    - 相対特徴量: 数値9列の (値 - レース内平均)  ← LightGBM 用に追加
    - レース特徴量: ~40次元
"""

from itertools import permutations

import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BOATS_PER_RACE  = 6
N_NUMERIC_FEATS = 10  # build_boat_features の最初 10 列が数値特徴量 (ST追加で9→10)


# -------------------------------------------------------
# 特徴量変換（per-boat-per-race フラット行列）
# -------------------------------------------------------

def build_X(boat_features: np.ndarray, race_features: np.ndarray) -> np.ndarray:
    """
    (n_races, 6, n_boat) + (n_races, n_race) → (n_races*6, n_feats)

    追加する相対特徴量: 数値9列の (boat_val - race_mean)
    これにより LightGBM が艇間の相対優劣を直接学べる。
    """
    n_races = boat_features.shape[0]

    # 相対特徴量: 数値9列の差分 (n_races, 6, 9)
    num_feats  = boat_features[:, :, :N_NUMERIC_FEATS]          # (n_races, 6, 9)
    race_mean  = num_feats.mean(axis=1, keepdims=True)           # (n_races, 1, 9)
    rel_feats  = (num_feats - race_mean).astype(np.float32)      # (n_races, 6, 9)

    # レース特徴量を 6 艇分に複製
    race_tiled = np.tile(race_features[:, np.newaxis, :], (1, 6, 1))  # (n_races, 6, R)

    # [艇特徴量 | 相対特徴量 | レース特徴量]
    combined = np.concatenate([boat_features, rel_feats, race_tiled], axis=2)
    return combined.reshape(n_races * 6, -1).astype(np.float32)


def build_y(positions: np.ndarray) -> np.ndarray:
    """
    着順 → relevance スコア (高いほど良い)。
    1着 → 5, 2着 → 4, ..., 6着 → 0
    """
    rel = (BOATS_PER_RACE + 1 - positions).astype(np.int32)
    return rel.reshape(-1)


# -------------------------------------------------------
# 学習
# -------------------------------------------------------

def train(train_data: tuple,
          val_data: tuple,
          n_estimators: int   = 3000,
          learning_rate: float = 0.05,
          num_leaves: int      = 63,
          min_child_samples: int = 20,
          early_stopping_rounds: int = 50,
          **extra_params) -> lgb.Booster:
    """
    LightGBM LambdaRank を学習する。

    Parameters
    ----------
    train_data : (X_boat_tr, X_race_tr, y_tr, pay_tr) または (X_boat_tr, X_race_tr, y_tr)
    val_data   : 同形式

    Returns
    -------
    booster : lgb.Booster
    """
    if len(train_data) == 4:
        X_boat_tr, X_race_tr, y_tr, pay_tr = train_data
        pay_pos  = pay_tr[pay_tr > 0]
        pay_mean = float(np.mean(pay_pos)) if len(pay_pos) > 0 else 1.0
        pay_med  = float(np.median(pay_pos)) if len(pay_pos) > 0 else 1.0

        # 改善: log1p スケールで重みを計算
        # 理由: sqrt より log はより高配当レースを重視しつつ外れ値を抑制する
        #       log(1 + pay/median) → 中央値のレースを基準 (weight≈0.69) に正規化
        w_tr = np.log1p(pay_tr / (pay_med + 1e-8)).astype(np.float32)
        # 有効レース以外 (pay=0) は weight=0 になるので 0.1 でクリップ
        w_tr     = np.clip(w_tr, 0.1, None)
        # 平均を 1.0 に正規化して元のスケール感を保つ
        w_tr     = w_tr / (w_tr.mean() + 1e-8)
        w_tr     = np.clip(w_tr, 0.1, 10.0)
        weight_tr  = np.repeat(w_tr, BOATS_PER_RACE)   # 同じレース内の全艇に同じ重み
        print(f"ペイアウト重み (log1p): mean={w_tr.mean():.2f}  "
              f"min={w_tr.min():.2f}  max={w_tr.max():.2f}  "
              f"pay_med={pay_med:.0f}円")
    else:
        X_boat_tr, X_race_tr, y_tr = train_data
        weight_tr = None

    if len(val_data) == 4:
        X_boat_va, X_race_va, y_va, _ = val_data
    else:
        X_boat_va, X_race_va, y_va = val_data

    X_tr = build_X(X_boat_tr, X_race_tr)
    X_va = build_X(X_boat_va, X_race_va)

    y_label_tr = build_y(y_tr)
    y_label_va = build_y(y_va)

    n_tr = len(X_boat_tr)
    n_va = len(X_boat_va)

    dtrain = lgb.Dataset(X_tr, label=y_label_tr,
                         group=[BOATS_PER_RACE] * n_tr,
                         weight=weight_tr)
    dval   = lgb.Dataset(X_va, label=y_label_va,
                         group=[BOATS_PER_RACE] * n_va,
                         reference=dtrain)

    params = {
        "objective":         "lambdarank",
        "metric":            "ndcg",
        "ndcg_eval_at":      [1, 3],
        "num_leaves":        num_leaves,
        "learning_rate":     learning_rate,
        "min_child_samples": min_child_samples,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l2":         0.1,
        "verbose":           -1,
    }
    # Optuna で最適化された全パラメータで上書き
    # (lambda_l1, lambda_l2, feature_fraction, bagging_fraction 等)
    if extra_params:
        params.update(extra_params)
        print(f"Optuna パラメータを適用: {extra_params}")

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # 特徴量重要度（上位15）
    imp      = booster.feature_importance(importance_type="gain")
    top_idx  = np.argsort(-imp)[:15]
    print("\n=== 特徴量重要度 Top15 (gain) ===")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. feature_{idx:3d}: {imp[idx]:.1f}")

    return booster


# -------------------------------------------------------
# 内部ユーティリティ
# -------------------------------------------------------

def _predict_scores(booster: lgb.Booster,
                    X_boat: np.ndarray,
                    X_race: np.ndarray) -> np.ndarray:
    """各艇のスコアを予測する。shape: (n_races, 6)"""
    X = build_X(X_boat, X_race)
    return booster.predict(X).reshape(-1, BOATS_PER_RACE)


def _plackett_luce_probs(scores: np.ndarray):
    """
    各レースの全120通りを Plackett-Luce 確率でスコアリングする。

    Returns
    -------
    top_probs  : (n_races,)  最高確率の値
    top_combos : (n_races,)  最高確率の組番文字列
    top4_lists : list of list  上位4通りの組番リスト
    """
    exp_s      = np.exp(scores - scores.max(axis=1, keepdims=True))
    n          = len(scores)
    top_probs  = np.zeros(n)
    top_combos = np.empty(n, dtype=object)
    top4_lists = []

    for i in range(n):
        s     = exp_s[i]
        total = s.sum()
        race_probs = []
        for a, b, c in permutations(range(6), 3):
            p = (s[a] / total) \
              * (s[b] / (total - s[a])) \
              * (s[c] / (total - s[a] - s[b]))
            race_probs.append((p, f"{a+1}-{b+1}-{c+1}"))
        race_probs.sort(reverse=True)

        top_probs[i]  = race_probs[0][0]
        top_combos[i] = race_probs[0][1]
        top4_lists.append([c for _, c in race_probs[:4]])

    return top_probs, top_combos, top4_lists


# -------------------------------------------------------
# Optuna ハイパーパラメータ最適化
# -------------------------------------------------------

def tune_hyperparams(train_data: tuple,
                     val_data: tuple,
                     n_trials: int = 50) -> dict:
    """
    Optuna で LightGBM LambdaRank のハイパーパラメータを最適化する。

    最適化対象: num_leaves, learning_rate, min_child_samples,
                lambda_l1, lambda_l2, feature_fraction, bagging_fraction

    評価指標: 検証データの NDCG@3 (最大化)

    Parameters
    ----------
    train_data : (X_boat_tr, X_race_tr, y_tr, pay_tr)
    val_data   : (X_boat_va, X_race_va, y_va, pay_va)
    n_trials   : Optuna の試行回数 (推奨: 50〜100)

    Returns
    -------
    best_params : dict  最適なハイパーパラメータ
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna がインストールされていません。\n"
            "pip install optuna でインストールしてください。"
        )

    X_boat_tr, X_race_tr, y_tr, pay_tr = train_data
    X_boat_va, X_race_va, y_va, _      = val_data

    X_tr = build_X(X_boat_tr, X_race_tr)
    X_va = build_X(X_boat_va, X_race_va)
    y_label_tr = build_y(y_tr)
    y_label_va = build_y(y_va)

    n_tr = len(X_boat_tr)
    n_va = len(X_boat_va)

    # ペイアウト重みを事前に計算
    if pay_tr is not None:
        pay_pos  = pay_tr[pay_tr > 0]
        pay_med  = float(np.median(pay_pos)) if len(pay_pos) > 0 else 1.0
        w_tr     = np.log1p(pay_tr / (pay_med + 1e-8)).astype(np.float32)
        w_tr     = np.clip(w_tr, 0.1, None)
        w_tr     = w_tr / (w_tr.mean() + 1e-8)
        w_tr     = np.clip(w_tr, 0.1, 10.0)
        weight_tr = np.repeat(w_tr, BOATS_PER_RACE)
    else:
        weight_tr = None

    dtrain = lgb.Dataset(X_tr, label=y_label_tr,
                         group=[BOATS_PER_RACE] * n_tr,
                         weight=weight_tr)
    dval   = lgb.Dataset(X_va, label=y_label_va,
                         group=[BOATS_PER_RACE] * n_va,
                         reference=dtrain)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective":          "lambdarank",
            "metric":             "ndcg",
            "ndcg_eval_at":       [3],
            "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples":  trial.suggest_int("min_child_samples", 5, 50),
            "lambda_l1":          trial.suggest_float("lambda_l1", 1e-4, 1.0, log=True),
            "lambda_l2":          trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
            "feature_fraction":   trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":       5,
            "feature_pre_filter": False,   # Optuna が min_data_in_leaf を変えても安全に動くよう無効化
            "verbose":            -1,
        }

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        # NDCG@3 の最終スコアを返す
        return booster.best_score["valid_0"]["ndcg@3"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print("\n=== Optuna 最適化結果 ===")
    print(f"  試行回数: {n_trials}")
    print(f"  最良 NDCG@3: {study.best_value:.6f}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    return best


# -------------------------------------------------------
# 評価
# -------------------------------------------------------

def evaluate(booster: lgb.Booster, test_data: tuple):
    """
    3連単の的中率・回収率を計算してグラフを表示する。

    Returns
    -------
    top1_hits, top1_gains, top1_rr, top4_hits, top4_gains, top4_rr
    """
    X_boat_te, X_race_te, _, y_tri_str_te, y_tri_pay_te = test_data
    scores     = _predict_scores(booster, X_boat_te, X_race_te)
    actual_str = np.array([s.replace(" ", "") for s in y_tri_str_te])
    payouts    = y_tri_pay_te.astype(float)

    _, _, top4_lists = _plackett_luce_probs(scores)

    top4_hits = np.array([actual_str[i] in top4_lists[i] for i in range(len(scores))])

    BET_PER_RACE = 100 * 4
    top4_gains   = np.where(top4_hits, payouts, 0.0)
    top4_rr      = top4_gains.sum() / (BET_PER_RACE * len(top4_gains)) * 100

    print("=" * 40)
    print("=== 3連単 評価（上位4通り購入）===")
    print("=" * 40)
    print(f"レース数      : {len(top4_hits):,}")
    print(f"的中率        : {top4_hits.mean()*100:.2f}%  (理論ランダム ~3.3%)")
    print(f"回収率        : {top4_rr:.2f}%  (損益分岐 100%)")
    print(f"的中回数      : {top4_hits.sum():,}")
    print(f"賭け金/レース : {BET_PER_RACE} 円（100円 × 4通り）")
    if top4_hits.sum() > 0:
        print(f"的中時平均払戻: {top4_gains[top4_gains > 0].mean():.0f} 円")
        print(f"最高払戻      : {top4_gains.max():.0f} 円")

    pred_order   = np.argsort(-scores, axis=1)
    pred_tri_str = np.array([f"{p[0]+1}-{p[1]+1}-{p[2]+1}" for p in pred_order[:, :3]])
    top1_hits    = pred_tri_str == actual_str
    top1_gains   = np.where(top1_hits, payouts, 0.0)
    top1_rr      = top1_gains.sum() / (100 * len(top1_gains)) * 100

    print()
    print("=== 参考：上位1通り購入 ===")
    print(f"的中率  : {top1_hits.mean()*100:.3f}%")
    print(f"回収率  : {top1_rr:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cumsum, label, rr, bet in [
        (axes[0], np.cumsum(top1_gains - 100),          "上位1通り", top1_rr, 100),
        (axes[1], np.cumsum(top4_gains - BET_PER_RACE), "上位4通り", top4_rr, BET_PER_RACE),
    ]:
        ax.plot(cumsum, label="累積収支")
        ax.axhline(0, color="red", linestyle="--", alpha=0.7, label="損益分岐")
        ax.set_xlabel("レース数")
        ax.set_ylabel("累積収支 (円)")
        ax.set_title(f"{label}  回収率: {rr:.2f}%  ({bet}円/レース)")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    return top1_hits, top1_gains, top1_rr, top4_hits, top4_gains, top4_rr


# -------------------------------------------------------
# 信頼度閾値分析
# -------------------------------------------------------

def evaluate_threshold(booster: lgb.Booster,
                       test_data: tuple,
                       thresholds: list = None):
    """
    信頼度閾値別の回収率を分析する（上位1通り購入）。

    信頼度の定義:
        1着予測艇の softmax 確率。
        ランダム相当 = 1/6 ≈ 17%。これを超えるほど予測に確信がある。

    Returns
    -------
    results : list of (threshold, n_bets, hit_rate_pct, return_rate_pct)
    """
    if thresholds is None:
        thresholds = [0.17, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    X_boat_te, X_race_te, _, y_tri_str_te, y_tri_pay_te = test_data
    scores     = _predict_scores(booster, X_boat_te, X_race_te)
    actual_str = np.array([s.replace(" ", "") for s in y_tri_str_te])
    payouts    = y_tri_pay_te.astype(float)

    # 信頼度 = 1着予測艇の softmax 確率（スケール不変）
    exp_s      = np.exp(scores - scores.max(axis=1, keepdims=True))
    boat_probs = exp_s / exp_s.sum(axis=1, keepdims=True)   # (n, 6)
    confidence = boat_probs.max(axis=1)                      # (n,)

    # 購入組み合わせは Plackett-Luce 上位1通り（変わらず）
    _, top_combos, _ = _plackett_luce_probs(scores)

    print("=" * 62)
    print("=== 信頼度閾値別 回収率（上位1通り・100円/レース）===")
    print("    信頼度 = 1着予測艇の softmax 確率  (ランダム≈17%)")
    print(f"{'閾値':>7}  {'賭けレース':>10}  {'的中率':>8}  {'回収率':>8}")
    print("-" * 62)

    results = []
    for thr in thresholds:
        mask = confidence >= thr
        if mask.sum() == 0:
            print(f"{thr:>7.1%}  {'—':>10}  {'—':>8}  {'—':>8}")
            continue
        hits  = top_combos[mask] == actual_str[mask]
        gains = np.where(hits, payouts[mask], 0.0)
        rr    = gains.sum() / (100 * mask.sum()) * 100
        results.append((thr, int(mask.sum()), float(hits.mean() * 100), float(rr)))
        print(f"{thr:>7.1%}  {mask.sum():>10,}  {hits.mean()*100:>7.2f}%  {rr:>7.2f}%")

    if results:
        thrs, counts, hit_rates, rrs = zip(*results)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot([t * 100 for t in thrs], rrs, marker="o")
        ax1.axhline(100, color="red", linestyle="--", label="損益分岐")
        ax1.axvline(100 / 6, color="gray", linestyle=":", alpha=0.6, label="ランダム相当 (17%)")
        ax1.set_xlabel("信頼度閾値 (%)"); ax1.set_ylabel("回収率 (%)")
        ax1.set_title("信頼度閾値 vs 回収率"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot([t * 100 for t in thrs], counts, marker="o", color="orange")
        ax2.set_xlabel("信頼度閾値 (%)"); ax2.set_ylabel("賭けレース数")
        ax2.set_title("信頼度閾値 vs 賭けレース数"); ax2.grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

    return results

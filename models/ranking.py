"""
Plackett-Luce ランキングネット — 3連単予測モデル。

モデル種別:
    build_model()           : 標準 MLP ランキングネット（ベースライン相当）
    build_attention_model() : Self-Attention ランキングネット（推奨）

改善点:
    1. Self-Attention: 6艇間の相互作用を MultiHeadAttention で学習
    2. ペイアウト重み付き学習: 高払戻レースを重視
    3. 信頼度閾値分析: 確信度の高いレースだけ購入して回収率を改善
"""

from itertools import permutations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from tensorflow.keras.layers import (
    Input, Dense, Dropout, TimeDistributed,
    RepeatVector, Concatenate, Reshape,
    MultiHeadAttention, LayerNormalization, Add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------------
# 損失関数
# -------------------------------------------------------

def plackett_luce_loss_top3(y_true, y_pred):
    """
    Plackett-Luce ランキング損失（上位3着のみ）。

    Parameters
    ----------
    y_true : (batch, 6)  各枠の着順（float, 1.0=1着 〜 6.0=6着）
    y_pred : (batch, 6)  各枠のスコア（ロジット）
    """
    y_pred     = y_pred - tf.stop_gradient(tf.reduce_max(y_pred, axis=-1, keepdims=True))
    exp_s      = tf.exp(y_pred)

    sorted_idx  = tf.argsort(tf.cast(y_true, tf.float32), axis=-1)
    batch_size  = tf.shape(y_true)[0]
    batch_range = tf.cast(tf.range(batch_size), tf.int32)

    total_loss = tf.zeros([batch_size], dtype=tf.float32)
    remaining  = tf.reduce_sum(exp_s, axis=-1)

    for k in range(3):
        boat_k     = sorted_idx[:, k]
        score_k    = tf.gather_nd(exp_s, tf.stack([batch_range, boat_k], axis=1))
        total_loss = total_loss - tf.math.log(score_k / (remaining + 1e-8) + 1e-8)
        remaining  = remaining - score_k

    return tf.reduce_mean(total_loss)


# -------------------------------------------------------
# モデル定義
# -------------------------------------------------------

def build_model(n_boat_feats: int,
                n_race_feats: int,
                units: tuple = (128, 64, 32),
                dropout: float = 0.2) -> Model:
    """標準 MLP ランキングネット（比較用）。"""
    boat_input = Input(shape=(6, n_boat_feats), name="boat_features")
    race_input = Input(shape=(n_race_feats,),   name="race_features")

    race_tiled = RepeatVector(6)(race_input)
    x = Concatenate(axis=-1)([boat_input, race_tiled])

    for u in units:
        x = TimeDistributed(Dense(u, activation="relu"))(x)
        x = TimeDistributed(Dropout(dropout))(x)

    scores = Reshape((6,))(TimeDistributed(Dense(1))(x))

    model = Model(inputs=[boat_input, race_input], outputs=scores, name="RankingNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=plackett_luce_loss_top3,
    )
    return model


def build_attention_model(n_boat_feats: int,
                          n_race_feats: int,
                          d_model: int   = 128,
                          heads: int     = 4,
                          n_layers: int  = 2,
                          dropout: float = 0.2) -> Model:
    """
    Self-Attention ランキングネット（推奨）。

    6艇の特徴量を d_model 次元に射影してから MultiHeadAttention で
    艇間の相互作用を学習し、各艇のスコアを出力する。

    Parameters
    ----------
    d_model  : 隠れ層次元数（Attention の key_dim = d_model // heads）
    heads    : Attention ヘッド数
    n_layers : Attention + FFN ブロックの積み重ね数
    """
    boat_input = Input(shape=(6, n_boat_feats), name="boat_features")
    race_input = Input(shape=(n_race_feats,),   name="race_features")

    # レース特徴量を 6 艇分に複製
    race_tiled = RepeatVector(6)(race_input)
    x = Concatenate(axis=-1)([boat_input, race_tiled])

    # d_model 次元に射影
    x = TimeDistributed(Dense(d_model, activation="relu"))(x)
    x = TimeDistributed(Dropout(dropout))(x)

    # Self-Attention + FFN ブロック × n_layers
    for _ in range(n_layers):
        # Self-Attention（艇間の相互作用）
        attn_out = MultiHeadAttention(
            num_heads=heads, key_dim=d_model // heads, dropout=dropout
        )(x, x)
        attn_out = Dropout(dropout)(attn_out)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))

        # Position-wise FFN
        ff = TimeDistributed(Dense(d_model * 2, activation="relu"))(x)
        ff = TimeDistributed(Dropout(dropout))(ff)
        ff = TimeDistributed(Dense(d_model))(ff)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))

    # 各艇のスコア (batch, 6)
    scores = Reshape((6,))(TimeDistributed(Dense(1))(x))

    model = Model(inputs=[boat_input, race_input], outputs=scores,
                  name="AttentionRankingNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=plackett_luce_loss_top3,
    )
    return model


# -------------------------------------------------------
# 学習
# -------------------------------------------------------

def train(model: Model,
          train_data: tuple,
          val_data:   tuple,
          epochs:     int   = 60,
          batch_size: int   = 256,
          patience:   int   = 8):
    """
    モデルを学習する。

    Parameters
    ----------
    train_data : (X_boat_tr, X_race_tr, y_tr)  または
                 (X_boat_tr, X_race_tr, y_tr, pay_tr)   ← ペイアウト重み付き
    val_data   : 同形式
    """
    if len(train_data) == 4:
        X_boat_tr, X_race_tr, y_tr, pay_tr = train_data
        pay_mean      = float(np.mean(pay_tr[pay_tr > 0])) if (pay_tr > 0).any() else 1.0
        sample_weight = np.clip(
            np.sqrt(pay_tr / (pay_mean + 1e-8)), 0.1, 10.0
        ).astype(np.float32)
        print(f"ペイアウト重み: mean={sample_weight.mean():.2f}  "
              f"min={sample_weight.min():.2f}  max={sample_weight.max():.2f}")
    else:
        X_boat_tr, X_race_tr, y_tr = train_data
        sample_weight = None

    if len(val_data) == 4:
        X_boat_va, X_race_va, y_va, _ = val_data
    else:
        X_boat_va, X_race_va, y_va = val_data

    cb = EarlyStopping(
        monitor="val_loss", patience=patience,
        restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        [X_boat_tr, X_race_tr], y_tr,
        validation_data=([X_boat_va, X_race_va], y_va),
        sample_weight=sample_weight,
        epochs=epochs, batch_size=batch_size, callbacks=[cb],
    )

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("Plackett-Luce loss")
    plt.title(f"{model.name} 学習曲線"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    return history


# -------------------------------------------------------
# 評価
# -------------------------------------------------------

def evaluate(model: Model, test_data: tuple):
    """
    3連単の的中率・回収率を計算してグラフを表示する。

    Returns
    -------
    top1_hits, top1_gains, top1_rr, top4_hits, top4_gains, top4_rr
    """
    X_boat_te, X_race_te, _, y_tri_str_te, y_tri_pay_te = test_data

    scores     = model.predict([X_boat_te, X_race_te], verbose=0)
    actual_str = np.array([s.replace(" ", "") for s in y_tri_str_te])
    payouts    = y_tri_pay_te.astype(float)

    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))

    top4_hits   = np.zeros(len(scores), dtype=bool)
    top4_combos = []

    for i in range(len(scores)):
        s     = exp_s[i]
        total = s.sum()
        race_probs = []
        for a, b, c in permutations(range(6), 3):
            p = (s[a] / total) \
              * (s[b] / (total - s[a])) \
              * (s[c] / (total - s[a] - s[b]))
            race_probs.append((p, f"{a+1}-{b+1}-{c+1}"))
        race_probs.sort(reverse=True)

        top4 = [combo for _, combo in race_probs[:4]]
        top4_combos.append(top4)
        top4_hits[i] = actual_str[i] in top4

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

def evaluate_threshold(model: Model,
                       test_data: tuple,
                       thresholds: list = None):
    """
    信頼度閾値を変えながら「確信度の高いレースだけ購入」の回収率を分析する。

    Parameters
    ----------
    thresholds : 最高確率の閾値リスト（デフォルト 0.0〜0.40）

    Returns
    -------
    results : list of (threshold, n_bets, hit_rate_pct, return_rate_pct)
    """
    if thresholds is None:
        thresholds = [0.17, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    X_boat_te, X_race_te, _, y_tri_str_te, y_tri_pay_te = test_data
    scores     = model.predict([X_boat_te, X_race_te], verbose=0)
    actual_str = np.array([s.replace(" ", "") for s in y_tri_str_te])
    payouts    = y_tri_pay_te.astype(float)

    # 信頼度 = 1着予測艇の softmax 確率（スケール不変、ランダム≈17%）
    exp_s      = np.exp(scores - scores.max(axis=1, keepdims=True))
    boat_probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    confidence = boat_probs.max(axis=1)

    # 購入組み合わせは Plackett-Luce 上位1通り
    top_combos = np.empty(len(scores), dtype=object)
    for i in range(len(scores)):
        s     = exp_s[i]
        total = s.sum()
        best_p = 0.0; best_c = ""
        for a, b, c in permutations(range(6), 3):
            p = (s[a] / total) \
              * (s[b] / (total - s[a])) \
              * (s[c] / (total - s[a] - s[b]))
            if p > best_p:
                best_p = p; best_c = f"{a+1}-{b+1}-{c+1}"
        top_combos[i] = best_c

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
        hits   = top_combos[mask] == actual_str[mask]
        gains  = np.where(hits, payouts[mask], 0.0)
        rr     = gains.sum() / (100 * mask.sum()) * 100
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

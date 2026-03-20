"""
単勝予測モデル（ベースライン）— 6 クラス分類。

既存アプローチ: 全6艇の特徴量を1ベクトルに並べ、
sparse_categorical_crossentropy で1着艇番を分類する。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


def build_model(n_features: int,
                units: tuple = (32, 16),
                activation: str = "tanh") -> Model:
    """
    単勝予測モデルを構築して返す。

    Parameters
    ----------
    n_features : 入力特徴量次元数
    units      : 隠れ層のユニット数（タプル）
    activation : 活性化関数
    """
    inputs = Input(shape=(n_features,))
    x = inputs
    for u in units:
        x = Dense(u, activation=activation)(x)
    outputs = Dense(6, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model


def train(model: Model,
          train_data: tuple,
          val_data:   tuple,
          epochs:     int = 20,
          batch_size: int = 64,
          patience:   int = 5):
    """
    Parameters
    ----------
    train_data : (X_tr, y_tr)  — y は艇番 1〜6（int）
    val_data   : (X_va, y_va)
    """
    X_tr, y_tr = train_data
    X_va, y_va = val_data

    cb = EarlyStopping(
        monitor="val_acc", mode="max", patience=patience,
        restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        X_tr, y_tr - 1,               # 0-indexed ラベルに変換
        validation_data=(X_va, y_va - 1),
        epochs=epochs, batch_size=batch_size, callbacks=[cb],
    )
    return history


def evaluate(model: Model, test_data: tuple):
    """
    単勝の的中率・回収率を計算する。

    Parameters
    ----------
    test_data : (X_te, y_boat_num, y_payout)

    Returns
    -------
    return_rate : float
    """
    X_te, y_boat_num, y_payout = test_data

    proba     = model.predict(X_te, verbose=0)
    y_pred    = np.argmax(proba, axis=1) + 1   # 1-indexed に戻す

    print(classification_report(y_boat_num, y_pred, digits=3))

    gains = np.where(y_pred == y_boat_num, y_payout.astype(float), 0.0)
    return_rate = gains.sum() / (100 * len(gains)) * 100
    print(f"単勝回収率: {return_rate:.2f}%")

    # 累積収支グラフ
    cumsum = np.cumsum(gains - 100)
    plt.figure(figsize=(10, 4))
    plt.plot(cumsum, label="累積収支")
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="損益分岐")
    plt.xlabel("レース数"); plt.ylabel("累積収支 (円)")
    plt.title(f"単勝ベースライン  回収率: {return_rate:.2f}%")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    return return_rate

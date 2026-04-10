"""
tests/conftest.py — 共通フィクスチャ。

DB を使うテストは tmp_db フィクスチャを使ってインメモリ SQLite を利用する。
recorder.DB_PATH をモンキーパッチして本番 DB を汚さない。
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest


# -------------------------------------------------------
# 一時 DB フィクスチャ
# -------------------------------------------------------

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """
    recorder.DB_PATH を一時ファイルに差し替えて init_db() を実行する。
    テスト終了後に自動削除される。
    """
    db_file = tmp_path / "test_auto.db"

    import auto.recorder as rec
    monkeypatch.setattr(rec, "DB_PATH", db_file)
    rec.init_db()
    yield db_file


# -------------------------------------------------------
# 共通テストデータ
# -------------------------------------------------------

@pytest.fixture
def sample_race():
    return {
        "race_id": "20260406_04_01",
        "jcd":     "04",
        "venue":   "平和島",
        "hd":      "20260406",
        "rno":     1,
        "stime":   "10:00",
    }


@pytest.fixture
def sample_probs():
    """120 通りの均一確率（合計=1）。"""
    p = np.ones(120) / 120
    return p


@pytest.fixture
def sample_odds():
    """すべてのコンボに オッズ 10.0 を設定（EV=確率×10-1）。"""
    from models.strategies import COMBO_STRS
    return {c: 10.0 for c in COMBO_STRS}


@pytest.fixture
def biased_probs():
    """
    1-2-3 に確率を集中させた偏りのある確率分布。
    EV チェックのためオッズを 20.0 に設定した場合に kelly が反応するはず。
    """
    from models.strategies import COMBO_IDX
    p = np.ones(120) * 0.001
    # 上位 5 通りに確率を集中
    hot = ["1-2-3", "1-2-4", "1-3-2", "2-1-3", "1-4-2"]
    for c in hot:
        p[COMBO_IDX[c]] = 0.08
    p = p / p.sum()
    return p


@pytest.fixture
def biased_odds(biased_probs):
    """上位コンボに高オッズ（EV > 0）を設定。"""
    from models.strategies import COMBO_STRS, COMBO_IDX
    odds = {c: 5.0 for c in COMBO_STRS}
    hot = ["1-2-3", "1-2-4", "1-3-2", "2-1-3", "1-4-2"]
    for c in hot:
        odds[c] = 20.0   # EV = 0.08×20 - 1 = 0.6  → 大きくプラス
    return odds

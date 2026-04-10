"""
tests/test_recorder.py — auto/recorder.py の単体テスト。

tmp_db フィクスチャで本番 DB を汚さずにインメモリ相当のテスト DB を使う。
"""

import sqlite3

import numpy as np
import pytest


# -------------------------------------------------------
# init_db
# -------------------------------------------------------

class TestInitDb:
    def test_tables_created(self, tmp_db):
        con = sqlite3.connect(tmp_db)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        con.close()
        assert "races" in tables
        assert "bets" in tables

    def test_idempotent(self, tmp_db):
        """2 回呼んでもエラーにならない。"""
        import auto.recorder as rec
        rec.init_db()  # 2 回目
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        con.close()
        assert count == 0


# -------------------------------------------------------
# upsert_race
# -------------------------------------------------------

class TestUpsertRace:
    def test_insert(self, tmp_db, sample_race):
        import auto.recorder as rec
        rec.upsert_race(sample_race)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT * FROM races WHERE race_id=?", (sample_race["race_id"],)).fetchone()
        con.close()
        assert row is not None

    def test_insert_ignore_duplicate(self, tmp_db, sample_race):
        import auto.recorder as rec
        rec.upsert_race(sample_race)
        rec.upsert_race(sample_race)  # 重複
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM races WHERE race_id=?", (sample_race["race_id"],)).fetchone()[0]
        con.close()
        assert count == 1


# -------------------------------------------------------
# save_strategy_bets
# -------------------------------------------------------

class TestSaveStrategyBets:
    def _setup_race(self, tmp_db, sample_race):
        import auto.recorder as rec
        rec.upsert_race(sample_race)

    def test_bets_saved(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        self._setup_race(tmp_db, sample_race)
        bets_dict = {"1-2-3": 300, "1-2-4": 200}
        rec.save_strategy_bets(sample_race["race_id"], "kelly", bets_dict, biased_odds)
        con = sqlite3.connect(tmp_db)
        rows = con.execute("SELECT * FROM bets WHERE race_id=?", (sample_race["race_id"],)).fetchall()
        con.close()
        assert len(rows) == 2

    def test_strategy_column_saved(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        self._setup_race(tmp_db, sample_race)
        rec.save_strategy_bets(sample_race["race_id"], "ip", {"1-2-3": 500}, biased_odds)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT strategy FROM bets WHERE race_id=?", (sample_race["race_id"],)).fetchone()
        con.close()
        assert row[0] == "ip"

    def test_ev_computed_when_probs_given(self, tmp_db, sample_race, biased_probs, biased_odds):
        import auto.recorder as rec
        from models.strategies import COMBO_IDX
        self._setup_race(tmp_db, sample_race)
        rec.save_strategy_bets(
            sample_race["race_id"], "kelly", {"1-2-3": 300}, biased_odds,
            probs_120=biased_probs,
        )
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT ev_at_bet, prob FROM bets WHERE race_id=?", (sample_race["race_id"],)).fetchone()
        con.close()
        assert row[0] is not None   # ev_at_bet が計算されている
        assert row[1] is not None   # prob が保存されている

    def test_pending_bets_replaced(self, tmp_db, sample_race, biased_odds):
        """同じ戦略を 2 回呼ぶと pending ベットが上書きされる。"""
        import auto.recorder as rec
        self._setup_race(tmp_db, sample_race)
        rec.save_strategy_bets(sample_race["race_id"], "kelly", {"1-2-3": 300}, biased_odds)
        rec.save_strategy_bets(sample_race["race_id"], "kelly", {"1-2-4": 500}, biased_odds)
        con = sqlite3.connect(tmp_db)
        rows = con.execute(
            "SELECT combo FROM bets WHERE race_id=? AND strategy='kelly'",
            (sample_race["race_id"],),
        ).fetchall()
        con.close()
        combos = [r[0] for r in rows]
        assert "1-2-4" in combos
        assert "1-2-3" not in combos  # 古いものは削除済み


# -------------------------------------------------------
# update_result
# -------------------------------------------------------

class TestUpdateResult:
    def _seed(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        rec.upsert_race(sample_race)
        # kelly: 1-2-3 に 300円、1-2-4 に 200円
        rec.save_strategy_bets(
            sample_race["race_id"], "kelly",
            {"1-2-3": 300, "1-2-4": 200}, biased_odds,
        )
        # ip: 1-2-3 に 500円
        rec.save_strategy_bets(
            sample_race["race_id"], "ip",
            {"1-2-3": 500}, biased_odds,
        )

    def test_win_bet_gets_payout(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        self._seed(tmp_db, sample_race, biased_odds)
        # 1-2-3 が的中、払戻 3530 円（100円賭けで）
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        con = sqlite3.connect(tmp_db)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM bets WHERE race_id=? AND combo='1-2-3'",
            (sample_race["race_id"],),
        ).fetchall()
        con.close()
        for row in rows:
            assert row["status"] == "win"
            assert row["payout"] > 0

    def test_lose_bet_zero_payout(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        self._seed(tmp_db, sample_race, biased_odds)
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        con = sqlite3.connect(tmp_db)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM bets WHERE race_id=? AND combo='1-2-4'",
            (sample_race["race_id"],),
        ).fetchone()
        con.close()
        assert row["status"] == "lose"
        assert row["payout"] == 0

    def test_payout_calculation(self, tmp_db, sample_race, biased_odds):
        """300円 × 払戻3530 / 100 = 10590円。"""
        import auto.recorder as rec
        self._seed(tmp_db, sample_race, biased_odds)
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        con = sqlite3.connect(tmp_db)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM bets WHERE race_id=? AND combo='1-2-3' AND strategy='kelly'",
            (sample_race["race_id"],),
        ).fetchone()
        con.close()
        assert row["payout"] == int(300 / 100 * 3530)  # 10590

    def test_return_dict_structure(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        self._seed(tmp_db, sample_race, biased_odds)
        result = rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        assert "total_bet" in result
        assert "total_return" in result
        assert "profit" in result
        assert result["profit"] == result["total_return"] - result["total_bet"]

    def test_race_result_stored(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        self._seed(tmp_db, sample_race, biased_odds)
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        con = sqlite3.connect(tmp_db)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT result_combo, result_payout FROM races WHERE race_id=?",
            (sample_race["race_id"],),
        ).fetchone()
        con.close()
        assert row["result_combo"] == "1-2-3"
        assert row["result_payout"] == 3530


# -------------------------------------------------------
# daily_summary
# -------------------------------------------------------

class TestDailySummary:
    def test_empty_day(self, tmp_db):
        import auto.recorder as rec
        s = rec.daily_summary("20260406")
        assert s["races_scheduled"] == 0
        assert s["total_bet"] == 0
        assert s["profit"] == 0

    def test_summary_after_result(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        rec.upsert_race(sample_race)
        rec.save_strategy_bets(
            sample_race["race_id"], "kelly",
            {"1-2-3": 300, "1-2-4": 200}, biased_odds,
        )
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        s = rec.daily_summary("20260406")
        assert s["total_bet"] == 500
        expected_return = int(300 / 100 * 3530)  # 10590
        assert s["total_return"] == expected_return
        assert s["hit_count"] == 1


# -------------------------------------------------------
# strategy_summary
# -------------------------------------------------------

class TestStrategySummary:
    def test_empty_returns_list(self, tmp_db):
        import auto.recorder as rec
        result = rec.strategy_summary("20260406")
        assert isinstance(result, list)
        assert result == []

    def test_two_strategies_returned(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        rec.upsert_race(sample_race)
        rec.save_strategy_bets(
            sample_race["race_id"], "kelly", {"1-2-3": 300}, biased_odds,
        )
        rec.save_strategy_bets(
            sample_race["race_id"], "ip", {"1-2-3": 500}, biased_odds,
        )
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        rows = rec.strategy_summary("20260406")
        strategies = {r["strategy"] for r in rows}
        assert "kelly" in strategies
        assert "ip" in strategies

    def test_strategy_profit_correct(self, tmp_db, sample_race, biased_odds):
        import auto.recorder as rec
        rec.upsert_race(sample_race)
        rec.save_strategy_bets(
            sample_race["race_id"], "kelly", {"1-2-3": 300}, biased_odds,
        )
        rec.update_result(sample_race["race_id"], "1-2-3", 3530)
        rows = rec.strategy_summary("20260406")
        kelly_row = next(r for r in rows if r["strategy"] == "kelly")
        assert kelly_row["total_bet"] == 300
        assert kelly_row["total_return"] == int(300 / 100 * 3530)
        assert kelly_row["profit"] == kelly_row["total_return"] - 300

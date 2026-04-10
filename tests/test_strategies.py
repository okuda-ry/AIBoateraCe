"""
tests/test_strategies.py — models/strategies.py の単体テスト。

スクレイピング・DB 不要。numpy と strategies.py だけに依存する。
"""

import numpy as np
import pytest

from models.strategies import (
    allocate_kelly,
    allocate_ip,
    run_all_strategies,
    COMBO_STRS,
    COMBO_IDX,
    _round_bets,
    _odds_array,
)


# -------------------------------------------------------
# _odds_array
# -------------------------------------------------------

class TestOddsArray:
    def test_shape(self, sample_odds):
        arr = _odds_array(sample_odds)
        assert arr.shape == (120,)

    def test_values(self, sample_odds):
        arr = _odds_array(sample_odds)
        assert np.all(arr == 10.0)

    def test_missing_combo_is_zero(self):
        arr = _odds_array({"1-2-3": 5.0})
        assert arr[COMBO_IDX["1-2-3"]] == 5.0
        assert arr[COMBO_IDX["1-2-4"]] == 0.0

    def test_empty_dict(self):
        arr = _odds_array({})
        assert arr.sum() == 0.0


# -------------------------------------------------------
# _round_bets
# -------------------------------------------------------

class TestRoundBets:
    def test_rounds_to_min_bet(self):
        amounts = np.array([150.0, 250.0] + [0.0] * 118)
        result = _round_bets(amounts, min_bet=100, budget=500)
        for v in result.values():
            assert v % 100 == 0

    def test_within_budget(self):
        amounts = np.ones(120) * 50.0   # 合計 6000 > budget=500
        result = _round_bets(amounts, min_bet=100, budget=500)
        assert sum(result.values()) <= 500

    def test_empty_when_all_below_min(self):
        amounts = np.ones(120) * 30.0
        result = _round_bets(amounts, min_bet=100, budget=1000)
        assert result == {}


# -------------------------------------------------------
# allocate_kelly
# -------------------------------------------------------

class TestAllocateKelly:
    def test_returns_dict(self, biased_probs, biased_odds):
        result = allocate_kelly(biased_probs, biased_odds, budget=1000)
        assert isinstance(result, dict)

    def test_no_bets_when_no_edge(self, sample_probs, sample_odds):
        """均一確率・オッズ10.0 → EV = 1/120 × 10 - 1 ≈ -0.917 → 全見送り。"""
        result = allocate_kelly(sample_probs, sample_odds, budget=1000, min_edge=0.05)
        assert result == {}

    def test_bets_when_high_ev(self, biased_probs, biased_odds):
        """上位コンボに高 EV があるとき kelly が反応して少なくとも1通りを選ぶ。"""
        result = allocate_kelly(biased_probs, biased_odds, budget=1000, min_edge=0.05)
        assert len(result) > 0

    def test_total_within_budget(self, biased_probs, biased_odds):
        budget = 1000
        result = allocate_kelly(biased_probs, biased_odds, budget=budget)
        assert sum(result.values()) <= budget

    def test_each_bet_multiple_of_min_bet(self, biased_probs, biased_odds):
        result = allocate_kelly(biased_probs, biased_odds, budget=1000, min_bet=100)
        for v in result.values():
            assert v % 100 == 0, f"{v} は min_bet(100) の倍数でない"

    def test_no_bets_when_no_odds(self, biased_probs):
        """オッズが全くない場合は空の辞書を返す。"""
        result = allocate_kelly(biased_probs, {}, budget=1000)
        assert result == {}

    def test_kelly_normalization_distributes_budget(self, biased_probs, biased_odds):
        """
        Kelly 正規化バグ修正の確認: kelly_f を正規化してから budget を掛けているので
        budget=1000 に対して合計が 100 円以上になることを確認。
        (旧バグでは kelly_f × budget が数円になり全て丸めで消えていた)
        """
        result = allocate_kelly(biased_probs, biased_odds, budget=1000, min_bet=100)
        if result:
            assert sum(result.values()) >= 100

    def test_combos_are_valid(self, biased_probs, biased_odds):
        """結果の全コンボが有効な 3 連単文字列か確認。"""
        result = allocate_kelly(biased_probs, biased_odds, budget=1000)
        for combo in result:
            assert combo in COMBO_IDX, f"無効なコンボ: {combo}"


# -------------------------------------------------------
# allocate_ip
# -------------------------------------------------------

class TestAllocateIP:
    def test_returns_dict(self, biased_probs, biased_odds):
        result = allocate_ip(biased_probs, biased_odds, budget=1000)
        assert isinstance(result, dict)

    def test_no_bets_when_no_edge(self, sample_probs, sample_odds):
        result = allocate_ip(sample_probs, sample_odds, budget=1000, min_edge=0.05)
        assert result == {}

    def test_bets_when_high_ev(self, biased_probs, biased_odds):
        result = allocate_ip(biased_probs, biased_odds, budget=1000, min_edge=0.05)
        assert len(result) > 0

    def test_total_within_budget(self, biased_probs, biased_odds):
        budget = 1000
        result = allocate_ip(biased_probs, biased_odds, budget=budget)
        assert sum(result.values()) <= budget

    def test_per_combo_cap(self, biased_probs, biased_odds):
        """
        IP バグ修正の確認: 1通りに全予算が集中しないこと。
        max_combos=5 なので各コンボは budget/5 = 200 円以下でなければならない。
        """
        budget = 1000
        max_combos = 5
        result = allocate_ip(biased_probs, biased_odds, budget=budget, max_combos=max_combos)
        per_combo_max = budget / max_combos
        for combo, amt in result.items():
            assert amt <= per_combo_max + 100, (  # +100 は丸め誤差マージン
                f"{combo} に {amt}円 (上限{per_combo_max}円) が割り当てられた"
            )

    def test_max_combos_respected(self, biased_probs, biased_odds):
        """max_combos を超える通り数は選ばない。"""
        result = allocate_ip(biased_probs, biased_odds, budget=1000, max_combos=3)
        assert len(result) <= 3

    def test_no_bets_when_no_odds(self, biased_probs):
        result = allocate_ip(biased_probs, {}, budget=1000)
        assert result == {}

    def test_each_bet_multiple_of_min_bet(self, biased_probs, biased_odds):
        result = allocate_ip(biased_probs, biased_odds, budget=1000, min_bet=100)
        for v in result.values():
            assert v % 100 == 0


# -------------------------------------------------------
# run_all_strategies
# -------------------------------------------------------

class TestRunAllStrategies:
    def test_returns_all_strategy_keys(self, biased_probs, biased_odds):
        result = run_all_strategies(biased_probs, biased_odds, budget=1000)
        assert "kelly" in result
        assert "ip" in result

    def test_each_value_is_dict(self, biased_probs, biased_odds):
        result = run_all_strategies(biased_probs, biased_odds, budget=1000)
        for k, v in result.items():
            assert isinstance(v, dict), f"{k} の結果が dict でない"

    def test_custom_strategies(self, biased_probs, biased_odds):
        """カスタム戦略辞書を渡せる。"""
        custom = {"only_kelly": allocate_kelly}
        result = run_all_strategies(biased_probs, biased_odds, budget=1000, strategies=custom)
        assert list(result.keys()) == ["only_kelly"]

    def test_strategy_error_does_not_crash(self, biased_probs, biased_odds):
        """戦略が例外を投げても run_all_strategies は落ちない。"""
        def bad_strategy(*a, **kw):
            raise ValueError("意図的なエラー")

        custom = {"bad": bad_strategy, "kelly": allocate_kelly}
        result = run_all_strategies(biased_probs, biased_odds, budget=1000, strategies=custom)
        assert result["bad"] == {}
        assert isinstance(result["kelly"], dict)


# -------------------------------------------------------
# COMBO_STRS / COMBO_IDX 一貫性
# -------------------------------------------------------

class TestComboConstants:
    def test_combo_count(self):
        assert len(COMBO_STRS) == 120   # 6P3 = 120

    def test_combo_idx_invertible(self):
        for i, c in enumerate(COMBO_STRS):
            assert COMBO_IDX[c] == i

    def test_no_duplicate_combos(self):
        assert len(set(COMBO_STRS)) == 120

    def test_combo_format(self):
        for c in COMBO_STRS:
            parts = c.split("-")
            assert len(parts) == 3
            assert all(p in "123456" for p in parts)
            assert len(set(parts)) == 3  # 3 艇全て異なる

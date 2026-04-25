import numpy as np

from models.kelly_betting import COMBO_INDEX, COMBO_STRS, value_bet_allocate


def test_value_bet_allocate_uses_budget_after_kelly_weighting():
    probs = np.ones(120) * 0.001
    hot = ["1-2-3", "1-2-4", "1-3-2"]
    for combo in hot:
        probs[COMBO_INDEX[combo]] = 0.08
    probs = probs / probs.sum()

    odds = {combo: 5.0 for combo in COMBO_STRS}
    for combo in hot:
        odds[combo] = 20.0

    result = value_bet_allocate(
        probs,
        odds,
        budget=1000,
        min_edge=0.05,
        kelly_frac=0.25,
        min_bet=100,
        max_combos=3,
    )

    assert result
    assert set(result).issubset(set(hot))
    assert sum(result.values()) <= 1000
    assert sum(result.values()) >= 900
    assert all(amount % 100 == 0 for amount in result.values())


def test_value_bet_allocate_returns_empty_without_edge():
    probs = np.ones(120) / 120
    odds = {combo: 10.0 for combo in COMBO_STRS}

    result = value_bet_allocate(probs, odds, budget=1000, min_edge=0.05)

    assert result == {}

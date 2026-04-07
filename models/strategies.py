"""
models/strategies.py — 購入意思決定戦略の統一インターフェース。

複数の戦略を同時に走らせて比較するための共通 API を定義する。

戦略一覧:
  - kelly    : 1/4 Kelly 規準（現行）
  - ip       : 整数計画法（scipy.optimize）
  - bayes    : ベイズ最適化パラメータ（Optuna で事前探索）

各戦略は同じシグネチャ allocate(probs, odds_dict, budget, **kwargs) -> dict を持つ。
"""

from __future__ import annotations

import numpy as np
from itertools import permutations
from typing import Callable

# -------------------------------------------------------
# 定数（kelly_betting.py と共有）
# -------------------------------------------------------
_COMBOS    = list(permutations(range(6), 3))
COMBO_STRS = [f"{a+1}-{b+1}-{c+1}" for a, b, c in _COMBOS]
COMBO_IDX  = {s: i for i, s in enumerate(COMBO_STRS)}


# -------------------------------------------------------
# 共通ユーティリティ
# -------------------------------------------------------

def _odds_array(odds_dict: dict) -> np.ndarray:
    """odds_dict → (120,) 配列"""
    arr = np.zeros(120)
    for i, c in enumerate(COMBO_STRS):
        if c in odds_dict:
            arr[i] = float(odds_dict[c])
    return arr


def _round_bets(amounts: np.ndarray, min_bet: int, budget: int) -> dict:
    """
    金額を min_bet 単位に丸めて予算内に収める。
    超過分は Kelly 配分の小さい順に削る。
    """
    amounts = (amounts / min_bet).round() * min_bet
    amounts = np.maximum(amounts, 0.0)
    # 超過を削る
    order = np.argsort(amounts)
    for i in order:
        if amounts.sum() <= budget:
            break
        if amounts[i] >= min_bet:
            amounts[i] -= min_bet
    return {COMBO_STRS[i]: int(amounts[i]) for i in range(120) if amounts[i] >= min_bet}


# -------------------------------------------------------
# 戦略1: Kelly 規準
# -------------------------------------------------------

def allocate_kelly(
    probs: np.ndarray,
    odds_dict: dict,
    budget: int      = 1000,
    min_edge: float  = 0.05,
    kelly_frac: float = 0.25,
    min_bet: int     = 100,
    top_n: int       = 10,
    **_,
) -> dict:
    """
    1/4 Kelly 規準。
    モデル確率上位 top_n 通りの中で EV > min_edge の組を Kelly 配分で購入。
    """
    odds_arr = _odds_array(odds_dict)

    # 上位 top_n に絞る
    top_idx = np.argsort(-probs)[:top_n]
    mask_top = np.zeros(120, dtype=bool)
    mask_top[top_idx] = True

    ev   = probs * odds_arr - 1.0
    mask = mask_top & (ev >= min_edge) & (odds_arr > 0)
    if mask.sum() == 0:
        return {}

    b       = np.where(odds_arr > 1, odds_arr - 1.0, 1e-9)
    kelly_f = np.where(mask, (probs - (1.0 - probs) / b) * kelly_frac, 0.0)
    kelly_f = np.maximum(kelly_f, 0.0)
    if kelly_f.sum() == 0:
        return {}

    # 3連単は確率が小さく kelly_f × budget が数円になって丸め後ゼロになる。
    # Kelly はどれに賭けるかを決め、比率で予算を分配する形に正規化する。
    kelly_f = kelly_f / kelly_f.sum()

    return _round_bets(budget * kelly_f, min_bet, budget)


# -------------------------------------------------------
# 戦略2: 整数計画法
# -------------------------------------------------------

def allocate_ip(
    probs: np.ndarray,
    odds_dict: dict,
    budget: int     = 1000,
    min_edge: float = 0.05,
    min_bet: int    = 100,
    max_combos: int = 5,
    top_n: int      = 20,
    **_,
) -> dict:
    """
    整数計画法による EV 最大化。

    目的関数: sum(ev[i] * x[i])  を最大化
    制約:
      - sum(x[i]) <= budget
      - x[i] in {0, min_bet, 2*min_bet, ...}
      - x[i] > 0 の数 <= max_combos
      - EV > min_edge の組のみ対象
      - 上位 top_n に絞る

    scipy が使えない場合は greedy フォールバック。
    """
    odds_arr = _odds_array(odds_dict)

    top_idx = np.argsort(-probs)[:top_n]
    ev      = probs * odds_arr - 1.0
    mask    = np.zeros(120, dtype=bool)
    mask[top_idx] = True
    mask   &= (ev >= min_edge) & (odds_arr > 0)

    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return {}

    # greedy: EV の高い順に max_combos まで選び、Kelly 比で配分
    ev_cands  = ev[candidates]
    sort_order = np.argsort(-ev_cands)
    selected  = candidates[sort_order[:max_combos]]

    # 各コンボの配分: EV 比例
    ev_sel = np.maximum(ev[selected], 1e-9)
    ratios = ev_sel / ev_sel.sum()
    amounts = np.zeros(120)
    amounts[selected] = ratios * budget

    result = _round_bets(amounts, min_bet, budget)
    if not result:
        return {}

    try:
        # scipy が使える場合は厳密解を求める
        from scipy.optimize import linprog

        n   = len(candidates)
        # linprog は最小化なので符号反転
        c   = -ev[candidates]
        # 予算制約: sum(x) <= budget
        A_ub = np.ones((1, n))
        b_ub = np.array([float(budget)])
        # 各変数の上限: 1通りへの集中を防ぐため budget/max_combos を上限とする
        per_combo_max = float(budget) / max_combos
        bounds = [(0.0, per_combo_max) for _ in range(n)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            amounts_lp = np.zeros(120)
            amounts_lp[candidates] = res.x
            lp_result = _round_bets(amounts_lp, min_bet, budget)
            if lp_result:
                result = lp_result
    except ImportError:
        pass  # greedy の結果をそのまま使う

    return result


# -------------------------------------------------------
# 戦略3: ベイズ最適化パラメータ（Optuna）
# -------------------------------------------------------

def build_bayes_strategy(
    history: list[dict],
    n_trials: int   = 50,
    budget: int     = 1000,
    min_bet: int    = 100,
) -> Callable:
    """
    過去の損益データから Kelly のパラメータを Optuna でベイズ最適化し、
    最適パラメータで固定した allocate 関数を返す。

    Parameters
    ----------
    history : [{"probs": np.ndarray, "odds_dict": dict,
                "result_combo": str, "result_payout": int}, ...]
              過去レースの予測・結果データ
    n_trials: Optuna のトライアル数

    Returns
    -------
    allocate_fn : allocate(probs, odds_dict, budget, **kwargs) -> dict
    """
    if not history:
        # 履歴がない場合はデフォルトパラメータで Kelly を返す
        print("[bayes] 履歴なし → デフォルトパラメータで Kelly を使用")
        return lambda probs, odds_dict, budget=budget, **kw: allocate_kelly(
            probs, odds_dict, budget=budget, **kw
        )

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[bayes] optuna 未インストール → デフォルト Kelly にフォールバック")
        return lambda probs, odds_dict, budget=budget, **kw: allocate_kelly(
            probs, odds_dict, budget=budget, **kw
        )

    def objective(trial: optuna.Trial) -> float:
        min_edge   = trial.suggest_float("min_edge",    0.0,  0.30)
        kelly_frac = trial.suggest_float("kelly_frac",  0.05, 0.50)
        top_n      = trial.suggest_int  ("top_n",       3,    20)

        total_bet    = 0
        total_return = 0.0

        for h in history:
            bets = allocate_kelly(
                h["probs"], h["odds_dict"],
                budget=budget, min_edge=min_edge,
                kelly_frac=kelly_frac, top_n=top_n,
                min_bet=min_bet,
            )
            if not bets:
                continue
            bet_sum = sum(bets.values())
            total_bet += bet_sum
            if h["result_combo"] in bets:
                payout = h["result_payout"]
                total_return += bets[h["result_combo"]] / 100 * payout

        if total_bet == 0:
            return 0.0
        return total_return / total_bet  # ROI を最大化

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"[bayes] 最適パラメータ: {best}  "
          f"(バックテスト ROI={study.best_value*100:.1f}%  {len(history)}レース)")

    def allocate_bayes(
        probs: np.ndarray,
        odds_dict: dict,
        budget: int = budget,
        **_,
    ) -> dict:
        return allocate_kelly(
            probs, odds_dict,
            budget     = budget,
            min_edge   = best["min_edge"],
            kelly_frac = best["kelly_frac"],
            top_n      = best["top_n"],
            min_bet    = min_bet,
        )

    return allocate_bayes


# -------------------------------------------------------
# 戦略レジストリ
# -------------------------------------------------------

# 全戦略の名前と allocate 関数のマッピング
# orchestrator から参照する
STRATEGIES: dict[str, Callable] = {
    "kelly": allocate_kelly,
    "ip":    allocate_ip,
    # "bayes" は build_bayes_strategy() で動的に生成して追加する
}


def run_all_strategies(
    probs: np.ndarray,
    odds_dict: dict,
    budget: int     = 1000,
    strategies: dict[str, Callable] | None = None,
    **kwargs,
) -> dict[str, dict]:
    """
    全戦略を実行してベット結果を返す。

    Returns
    -------
    {
        "kelly": {"1-3-5": 300, "2-1-4": 100},
        "ip":    {"1-3-5": 500},
        "bayes": {},
        ...
    }
    """
    if strategies is None:
        strategies = STRATEGIES

    results = {}
    for name, fn in strategies.items():
        try:
            bets = fn(probs, odds_dict, budget=budget, **kwargs)
            results[name] = bets
            total = sum(bets.values())
            n     = len(bets)
            status = f"{n}通り / {total:,}円" if bets else "見送り"
            print(f"  [{name:8s}] {status}")
        except Exception as e:
            print(f"  [{name:8s}] エラー: {e}")
            results[name] = {}
    return results

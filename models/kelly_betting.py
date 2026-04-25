"""
掛け金最適化モジュール — 1 レース N 円の予算配分。

======================================================
方式          オッズ   説明
======================================================
proportional  不要     予測確率に比例して配分（実運用可）
kelly_oracle  必要*    払戻を既知として Kelly 配分
             *バックテスト専用。実運用では事前オッズが必要。
======================================================

Kelly 規準の基本:
    f* = (b×p - q) / b = p - (1-p)/b
    b = 純オッズ = payout/100 - 1
    正の f* を持つ組み合わせにのみ賭ける。

比例配分は「全組み合わせのオッズが等しい」場合の Kelly と等価。
オッズなしで使える現実的な近似。
"""

from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 120 通りの組み合わせ
_COMBOS     = list(permutations(range(6), 3))          # (a,b,c) インデックス
COMBO_STRS  = [f"{a+1}-{b+1}-{c+1}" for a, b, c in _COMBOS]   # 文字列
COMBO_INDEX = {s: i for i, s in enumerate(COMBO_STRS)}          # 逆引き辞書

RANDOM_PROB = 1.0 / 120.0   # ランダム予測の確率 ≈ 0.83%


# -------------------------------------------------------
# 確率計算
# -------------------------------------------------------

def plackett_luce_probs(scores: np.ndarray) -> np.ndarray:
    """
    1 レース分のスコア (6,) → 120 通りの PL 確率 (120,)
    """
    exp_s = np.exp(scores - scores.max())
    total = exp_s.sum()
    probs = np.empty(120)
    for idx, (a, b, c) in enumerate(_COMBOS):
        probs[idx] = (
            (exp_s[a] / total)
            * (exp_s[b] / (total - exp_s[a]))
            * (exp_s[c] / (total - exp_s[a] - exp_s[b]))
        )
    return probs


# -------------------------------------------------------
# 配分関数
# -------------------------------------------------------

def proportional_allocate(probs: np.ndarray,
                          budget: int   = 1000,
                          min_prob_mul: float = 2.0,
                          min_bet: int  = 100) -> dict:
    """
    確率に比例して予算を配分する（オッズ不要）。

    Parameters
    ----------
    probs        : (120,) 各組み合わせの予測確率
    budget       : 総予算 (円)
    min_prob_mul : ランダム確率の何倍以上を購入対象にするか
                   2.0 → P > 2/120 ≈ 1.67% の組み合わせのみ
    min_bet      : 最低賭け金 (円、100 円単位)

    Returns
    -------
    bets : {combo_str: yen}  — 空 dict = 見送り
    """
    threshold = min_prob_mul * RANDOM_PROB
    mask      = probs >= threshold
    if mask.sum() == 0:
        return {}

    sel_probs = probs * mask
    raw       = budget * sel_probs / sel_probs.sum()

    # 100 円単位に丸め
    amounts = (raw / min_bet).round() * min_bet
    amounts = np.where(mask, np.maximum(amounts, min_bet), 0.0)

    # 予算超過を確率の低い順に削る
    order = np.argsort(probs)
    for i in order:
        if amounts.sum() <= budget:
            break
        if amounts[i] > 0:
            amounts[i] -= min_bet

    return {COMBO_STRS[i]: int(amounts[i]) for i in range(120) if amounts[i] >= min_bet}


def kelly_allocate(probs: np.ndarray,
                   payouts: np.ndarray,
                   budget: int   = 1000,
                   frac: float   = 0.25,
                   min_bet: int  = 100) -> dict:
    """
    Kelly 規準で配分する（オッズ既知の場合のみ使用可）。

    Parameters
    ----------
    probs   : (120,) 各組み合わせの予測確率
    payouts : (120,) 各組み合わせの払戻金 (100 円賭け時)
    frac    : Kelly 分数 (0.25 = 1/4 Kelly 推奨。フル Kelly は高分散)

    Returns
    -------
    bets : {combo_str: yen}  — 空 dict = 正 EV なし → 見送り
    """
    b  = payouts / 100.0 - 1.0              # 純オッズ
    q  = 1.0 - probs
    ev = probs * (payouts / 100.0) - 1.0    # EV per 100 yen

    # Kelly 配分 (正 EV のみ)
    kelly_f = np.where(ev > 0, probs - q / (b + 1e-9), 0.0)
    kelly_f = np.maximum(kelly_f, 0.0) * frac

    if kelly_f.sum() == 0:
        return {}

    # 合計が 1 超なら正規化 (過剰な賭けを防ぐ)
    if kelly_f.sum() > 1.0:
        kelly_f = kelly_f / kelly_f.sum()

    raw     = budget * kelly_f
    amounts = (raw / min_bet).round() * min_bet
    amounts = np.maximum(amounts, 0.0)

    # 予算超過を Kelly 配分の小さい順に削る
    order = np.argsort(kelly_f)
    for i in order:
        if amounts.sum() <= budget:
            break
        if amounts[i] >= min_bet:
            amounts[i] -= min_bet

    return {COMBO_STRS[i]: int(amounts[i]) for i in range(120) if amounts[i] >= min_bet}


# -------------------------------------------------------
# バックテスト評価
# -------------------------------------------------------

def evaluate(scores_all: np.ndarray,
             actual_strs: np.ndarray,
             payouts_all: np.ndarray,
             budget: int       = 1000,
             strategy: str     = "proportional",
             min_prob_mul: float = 2.0,
             kelly_frac: float  = 0.25):
    """
    全テストレースで掛け金配分戦略を評価する。

    Parameters
    ----------
    scores_all  : (n_races, 6)  モデルのスコア
    actual_strs : (n_races,)    実際の3連単組番
    payouts_all : (n_races,)    実際の3連単払戻金
    strategy    : "proportional" または "kelly_oracle"
                  kelly_oracle はバックテスト専用（実際の払戻を使用）

    Returns
    -------
    return_rate : float (%)
    """
    n = len(scores_all)

    total_bet    = 0
    total_return = 0.0
    skipped      = 0
    hit_races    = 0

    per_race_bet  = np.zeros(n)
    per_race_gain = np.zeros(n)

    for i in range(n):
        probs  = plackett_luce_probs(scores_all[i])
        actual = actual_strs[i].replace(" ", "")
        payout = float(payouts_all[i])

        if strategy == "proportional":
            bets = proportional_allocate(probs, budget=budget,
                                         min_prob_mul=min_prob_mul)

        elif strategy == "kelly_oracle":
            # oracle: 的中組み合わせは実際の払戻、外れは市場平均を仮定
            # 注意: バックテスト専用。実運用では締切前オッズが必要。
            est_payouts = 0.75 * 100.0 / (probs + 1e-12)   # モデル確率から推定
            win_idx     = COMBO_INDEX.get(actual)
            if win_idx is not None:
                est_payouts[win_idx] = payout               # 的中組は実績値で上書き
            bets = kelly_allocate(probs, est_payouts, budget=budget,
                                  frac=kelly_frac)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if not bets:
            skipped += 1
            continue

        bet_total = sum(bets.values())
        win_bet   = bets.get(actual, 0)
        gain      = (win_bet / 100.0) * payout if win_bet > 0 else 0.0

        total_bet    += bet_total
        total_return += gain
        per_race_bet[i]  = bet_total
        per_race_gain[i] = gain

        if gain > 0:
            hit_races += 1

    active   = n - skipped
    rr       = total_return / (total_bet + 1e-8) * 100
    hit_rate = hit_races / max(active, 1) * 100

    print("=" * 52)
    print(f"=== 掛け金最適化バックテスト [{strategy}] ===")
    print("=" * 52)
    print(f"総レース数          : {n:,}")
    print(f"賭けたレース        : {active:,}  (見送り: {skipped:,})")
    print(f"的中レース          : {hit_races:,}  ({hit_rate:.2f}%)")
    print(f"総賭け金            : {total_bet:,.0f} 円")
    print(f"総払戻              : {total_return:,.0f} 円")
    print(f"損益                : {total_return - total_bet:+,.0f} 円")
    print(f"回収率              : {rr:.2f}%  (損益分岐 100%)")
    if active > 0:
        avg_bet = per_race_bet[per_race_bet > 0].mean()
        print(f"平均賭け金/レース   : {avg_bet:.0f} 円")

    # --- 比較: フラットベッティングとの差 ---
    flat_bet    = active * budget
    flat_return = sum(
        budget / 100.0 * float(payouts_all[i])
        if actual_strs[i].replace(" ", "") == COMBO_STRS[
            int(np.argsort(-plackett_luce_probs(scores_all[i]))[0])
        ] else 0.0
        for i in range(n) if per_race_bet[i] > 0 or skipped == 0
    )

    # 累積収支グラフ
    profit   = per_race_gain - per_race_bet
    cumprofit = np.cumsum(profit)

    plt.figure(figsize=(10, 4))
    plt.plot(cumprofit, label="累積収支")
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="損益分岐")
    plt.xlabel("レース数"); plt.ylabel("累積収支 (円)")
    plt.title(f"掛け金最適化 [{strategy}]  回収率: {rr:.2f}%  "
              f"(予算 {budget} 円/レース)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    return rr


def value_bet_allocate(probs: np.ndarray,
                       odds_dict: dict,
                       budget: int   = 1000,
                       min_edge: float = 0.05,
                       kelly_frac: float = 0.25,
                       min_bet: int  = 100,
                       max_combos: int | None = None) -> dict:
    """
    期待値が正の組み合わせにのみ賭けるバリューベッティング。

    EV = prob × odds - 1  (odds は払戻倍率、例: 18.5)
    EV > min_edge のものだけ購入対象とし、Fractional Kelly の強さで配分する。

    Parameters
    ----------
    probs       : (120,) 各組み合わせの予測確率
    odds_dict   : {combo_str: float}  例: {"1-2-3": 18.5}
                  スクレイピングで取得した払戻倍率
    min_edge    : 最低エッジ (0.05 = EV が 5% 超の組のみ購入)
    kelly_frac  : Kelly 分数 (0.25 推奨。1.0 はフル Kelly で高分散)

    Returns
    -------
    bets : {combo_str: yen}  — 空 dict = 正 EV なし → 見送り
    """
    # odds_dict → (120,) 配列に変換
    odds_arr = np.zeros(120)
    for i, combo in enumerate(COMBO_STRS):
        if combo in odds_dict:
            odds_arr[i] = float(odds_dict[combo])

    # EV計算: prob × odds - 1  (EV > 0 → 期待値プラス)
    ev = probs * odds_arr - 1.0

    # 最低エッジ以上 & オッズ取得済み のものだけ購入対象
    mask = (ev >= min_edge) & (odds_arr > 0)
    if mask.sum() == 0:
        return {}

    # 1/4 Kelly: f = (EV / odds) × kelly_frac
    # Kelly の公式: f* = (b*p - (1-p)) / b  where b = odds - 1
    b       = np.where(odds_arr > 1, odds_arr - 1.0, 1e-9)
    kelly_f = np.where(mask, (probs - (1.0 - probs) / b) * kelly_frac, 0.0)
    kelly_f = np.maximum(kelly_f, 0.0)

    if kelly_f.sum() == 0:
        return {}

    if max_combos is not None and max_combos > 0:
        selected = np.argsort(-kelly_f)[:max_combos]
        keep = np.zeros(120, dtype=bool)
        keep[selected] = True
        kelly_f = np.where(keep, kelly_f, 0.0)

    if kelly_f.sum() == 0:
        return {}

    # 3連単は確率が小さく、素の Kelly 量だと100円単位に丸める前に消えやすい。
    # ここでは Kelly を「候補の強さ」として使い、買うと決めた候補へ予算を配分する。
    weights = kelly_f / kelly_f.sum()

    raw     = budget * weights
    amounts = (raw / min_bet).round() * min_bet
    amounts = np.maximum(amounts, 0.0)

    # 予算超過を Kelly 配分の小さい順に削る
    for i in np.argsort(weights):
        if amounts.sum() <= budget:
            break
        if amounts[i] >= min_bet:
            amounts[i] -= min_bet

    # Kelly 計算で全て min_bet 未満に丸まった場合（低確率・高オッズの組み合わせ）
    # → EV上位の組に min_bet を均等に配分するフォールバック
    if amounts.sum() == 0 and mask.sum() > 0:
        ev_vals = np.where(mask, probs * odds_arr - 1.0, -np.inf)
        top_n   = min(int(budget // min_bet), int(mask.sum()))
        top_idx = np.argsort(-ev_vals)[:top_n]
        for i in top_idx:
            if amounts.sum() + min_bet <= budget:
                amounts[i] = min_bet

    result = {COMBO_STRS[i]: int(amounts[i]) for i in range(120) if amounts[i] >= min_bet}

    if result:
        purchased_ev = [ev[i] for i, c in enumerate(COMBO_STRS) if c in result]
        print(f"[kelly] バリューベット: {len(result)}通り  "
              f"平均EV={np.mean(purchased_ev)*100:.1f}%  "
              f"合計={sum(result.values())}円")
    return result


def compute_ev_table(probs: np.ndarray, odds_dict: dict) -> list[dict]:
    """
    全組み合わせの期待値テーブルを返す（表示・デバッグ用）。

    Returns
    -------
    rows : [{"combo": str, "prob_pct": float, "odds": float, "ev_pct": float}, ...]
           EV降順でソート済み
    """
    rows = []
    for i, combo in enumerate(COMBO_STRS):
        odds = float(odds_dict.get(combo, 0.0))
        if odds <= 0:
            continue
        ev = probs[i] * odds - 1.0
        rows.append({
            "combo":   combo,
            "prob_pct": round(probs[i] * 100, 2),
            "odds":    odds,
            "ev_pct":  round(ev * 100, 1),
        })
    rows.sort(key=lambda x: -x["ev_pct"])
    return rows


def compare_strategies(scores_all: np.ndarray,
                       actual_strs: np.ndarray,
                       payouts_all: np.ndarray,
                       budget: int = 1000):
    """
    2 つの戦略を並べて比較する。
    """
    print("\n" + "=" * 52)
    print("  掛け金戦略 比較")
    print("=" * 52)

    rr_prop  = evaluate(scores_all, actual_strs, payouts_all,
                        budget=budget, strategy="proportional")
    rr_kelly = evaluate(scores_all, actual_strs, payouts_all,
                        budget=budget, strategy="kelly_oracle")

    print("\n" + "=" * 52)
    print("  戦略まとめ")
    print("=" * 52)
    print(f"  比例配分 (実運用可)   : {rr_prop:.2f}%")
    print(f"  Kelly oracle (上限)  : {rr_kelly:.2f}%")
    print()
    print("  ※ Kelly oracle はバックテスト専用。")
    print("    実運用には締切前のリアルタイムオッズが必要。")

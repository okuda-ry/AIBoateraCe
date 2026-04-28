"""
monitor.py — ドライランモニタリング Web UI（Flask Blueprint）。

既存の app.py に register_blueprint で追加して使う。
DB は data/auto.db を読み取り専用で参照する。
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

from flask import Blueprint, render_template, request

from auto.recorder import DB_PATH, init_db, daily_summary, strategy_summary
from models.strategies import STRATEGIES

monitor_bp = Blueprint("monitor", __name__, url_prefix="/monitor")

_db_initialized = False

@monitor_bp.before_request
def _ensure_db():
    """DB ファイルが存在する場合、テーブルを初期化する（冪等）。"""
    global _db_initialized
    if not _db_initialized and DB_PATH.exists():
        init_db()
        _db_initialized = True


# -------------------------------------------------------
# ユーティリティ
# -------------------------------------------------------

def _conn():
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _hd_param(default: str | None = None) -> str:
    """リクエストの ?hd= を取得。なければ今日。"""
    hd = request.args.get("hd", default)
    if not hd:
        hd = date.today().strftime("%Y%m%d")
    return hd


def _hd_display(hd: str) -> str:
    """YYYYMMDD → YYYY/MM/DD"""
    return f"{hd[:4]}/{hd[4:6]}/{hd[6:8]}"


def _profit_class(profit: int) -> str:
    if profit > 0:
        return "text-profit"
    if profit < 0:
        return "text-loss"
    return "text-muted"


_PREFERRED_STRATEGY_ORDER = [
    "kelly", "ip", "strict_flat", "true_kelly_cap",
    "dutch_value", "ip_conservative", "edge_band_flat",
    "favorite_overlay_flat", "tail_value_probe",
    "rank_1_23_45", "rank_123_box",
]


def _ordered_strategy_names(
    found: set[str] | None = None,
    include_configured: bool = False,
) -> list[str]:
    """表示用の戦略順。既知の戦略を先に、未知の戦略を後ろに並べる。"""
    names = set(found or set())
    if include_configured:
        names.update(STRATEGIES.keys())

    ordered = [s for s in _PREFERRED_STRATEGY_ORDER if s in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def _include_configured_strategy_rows(rows: list[dict]) -> list[dict]:
    """Add zero-value rows for configured strategies that have no DB records yet."""
    row_map = {r["strategy"]: r for r in rows}
    ordered = _ordered_strategy_names(set(row_map), include_configured=True)
    result = []
    for strategy in ordered:
        result.append(row_map.get(strategy, {
            "strategy": strategy,
            "races_bet": 0,
            "total_bet": 0,
            "total_return": 0,
            "profit": 0,
            "roi_pct": 0.0,
            "hit_count": 0,
            "hit_rate_pct": 0.0,
        }))
    return result


def _dashboard_metrics(summary: dict | None, strategies: list[dict]) -> dict:
    """Build derived dashboard metrics for quick dry-run evaluation."""
    if not summary:
        return {}

    settled = [s for s in strategies if s.get("total_bet", 0) > 0]
    best_roi = max(settled, key=lambda s: (s["roi_pct"], s["profit"]), default=None)
    best_profit = max(settled, key=lambda s: (s["profit"], s["roi_pct"]), default=None)
    worst_profit = min(settled, key=lambda s: (s["profit"], s["roi_pct"]), default=None)
    positive_count = sum(1 for s in settled if s["profit"] > 0)

    total_bet = summary.get("total_bet", 0) or 0
    races_predicted = summary.get("races_predicted", 0) or 0
    races_bet = summary.get("races_bet", 0) or 0
    hit_count = summary.get("hit_count", 0) or 0

    enriched = []
    for s in strategies:
        bet = s.get("total_bet", 0) or 0
        races = s.get("races_bet", 0) or 0
        ret = s.get("total_return", 0) or 0
        enriched.append({
            **s,
            "avg_bet_per_race": round(bet / races) if races else 0,
            "profit_per_race": round(s.get("profit", 0) / races) if races else 0,
            "return_per_hit": round(ret / s.get("hit_count", 0)) if s.get("hit_count", 0) else 0,
            "exposure_pct": round(bet / total_bet * 100, 1) if total_bet else 0.0,
        })

    return {
        "best_roi": best_roi,
        "best_profit": best_profit,
        "worst_profit": worst_profit,
        "positive_count": positive_count,
        "strategy_count": len(strategies),
        "active_strategy_count": len(settled),
        "bet_rate_pct": round(races_bet / races_predicted * 100, 1) if races_predicted else 0.0,
        "skip_rate_pct": round(summary.get("races_skip", 0) / races_predicted * 100, 1) if races_predicted else 0.0,
        "avg_bet_per_race": round(total_bet / races_bet) if races_bet else 0,
        "return_per_hit": round(summary.get("total_return", 0) / hit_count) if hit_count else 0,
        "roi_gap": round((best_roi["roi_pct"] - worst_profit["roi_pct"]), 1)
                   if best_roi and worst_profit else 0.0,
        "strategies": enriched,
    }


# -------------------------------------------------------
# ダッシュボード  GET /monitor
# -------------------------------------------------------

@monitor_bp.route("/")
def dashboard():
    hd = _hd_param()

    # 利用可能な日付リスト（DB にレースがある日）
    dates = []
    con = _conn()
    if con:
        rows = con.execute(
            "SELECT DISTINCT hd FROM races ORDER BY hd DESC LIMIT 30"
        ).fetchall()
        con.close()
        dates = [r["hd"] for r in rows]

    if hd not in dates and dates:
        hd = dates[0]

    has_db = DB_PATH.exists()
    summary = daily_summary(hd) if has_db else None
    strategies = _include_configured_strategy_rows(strategy_summary(hd)) if has_db else []
    metrics = _dashboard_metrics(summary, strategies)

    # レース一覧（ステータス集計用）
    races = []
    if con := _conn():
        rows = con.execute(
            """
            SELECT r.race_id, r.venue, r.rno, r.stime,
                   r.result_combo, r.result_payout,
                   r.predicted_at,
                   COUNT(b.id) AS n_bets,
                   COUNT(CASE WHEN b.status='win' THEN 1 END) AS win_count
            FROM races r
            LEFT JOIN bets b ON b.race_id = r.race_id
            WHERE r.hd = ?
            GROUP BY r.race_id
            ORDER BY r.stime, r.jcd, r.rno
            """,
            (hd,),
        ).fetchall()
        con.close()
        races = [dict(r) for r in rows]

    return render_template(
        "monitor/dashboard.html",
        hd=hd,
        hd_display=_hd_display(hd),
        dates=dates,
        summary=summary,
        metrics=metrics,
        strategies=strategies,
        races=races,
        has_db=has_db,
        profit_class=_profit_class,
    )


# -------------------------------------------------------
# レース一覧  GET /monitor/races?hd=YYYYMMDD
# -------------------------------------------------------

@monitor_bp.route("/races")
def races():
    hd = _hd_param()
    has_db = DB_PATH.exists()
    rows_out = []
    strategy_names = _ordered_strategy_names(include_configured=True)

    if has_db:
        con = _conn()
        if con is None:
            return render_template("monitor/races.html", hd=hd, hd_display=_hd_display(hd),
                                   races=[], strategy_names=strategy_names,
                                   has_db=False, profit_class=_profit_class)
        rows = con.execute(
            """
            SELECT
                r.race_id, r.venue, r.rno, r.stime,
                r.predicted_at, r.result_combo, r.result_payout,
                -- ステータス
                COUNT(CASE WHEN b.status='win'     THEN 1 END) AS win_count,
                COUNT(CASE WHEN b.status='pending' THEN 1 END) AS pending_count
            FROM races r
            LEFT JOIN bets b ON b.race_id = r.race_id
            WHERE r.hd = ?
            GROUP BY r.race_id
            ORDER BY r.stime, r.jcd, r.rno
            """,
            (hd,),
        ).fetchall()

        bet_rows = con.execute(
            """
            SELECT
                b.race_id,
                b.strategy,
                COALESCE(SUM(b.amount), 0) AS total_bet,
                COALESCE(SUM(b.payout), 0) AS total_return
            FROM bets b
            JOIN races r ON r.race_id = b.race_id
            WHERE r.hd = ?
            GROUP BY b.race_id, b.strategy
            ORDER BY b.strategy
            """,
            (hd,),
        ).fetchall()
        con.close()

        from collections import defaultdict
        race_strategy_map: dict[str, dict[str, dict]] = defaultdict(dict)
        found_strategies = set()
        for b in bet_rows:
            strategy = b["strategy"]
            found_strategies.add(strategy)
            total_bet = b["total_bet"]
            total_return = b["total_return"]
            race_strategy_map[b["race_id"]][strategy] = {
                "total_bet": total_bet,
                "total_return": total_return,
                "profit": total_return - total_bet,
            }

        strategy_names = _ordered_strategy_names(
            found_strategies,
            include_configured=True,
        )

        for r in rows:
            if r["predicted_at"] is None:
                status = "未予測"
                status_cls = "status-pending"
            elif r["result_combo"] is not None:
                status = "結果確定"
                status_cls = "status-done"
            elif r["pending_count"] > 0:
                status = "予測済み"
                status_cls = "status-predicted"
            else:
                status = "見送り"
                status_cls = "status-skip"

            strategies = {}
            for strategy in strategy_names:
                strategies[strategy] = race_strategy_map[r["race_id"]].get(
                    strategy,
                    {"total_bet": 0, "total_return": 0, "profit": 0},
                )

            rows_out.append({
                "race_id":       r["race_id"],
                "venue":         r["venue"],
                "rno":           r["rno"],
                "stime":         r["stime"],
                "status":        status,
                "status_cls":    status_cls,
                "result_combo":  r["result_combo"],
                "result_payout": r["result_payout"],
                "strategies":    strategies,
                "win_count":     r["win_count"],
            })

    return render_template(
        "monitor/races.html",
        hd=hd,
        hd_display=_hd_display(hd),
        races=rows_out,
        strategy_names=strategy_names,
        has_db=has_db,
        profit_class=_profit_class,
    )


# -------------------------------------------------------
# レース詳細  GET /monitor/race/<race_id>
# -------------------------------------------------------

@monitor_bp.route("/race/<race_id>")
def race_detail(race_id: str):
    has_db = DB_PATH.exists()
    race = None
    bets_by_strategy: dict[str, list] = {}

    if has_db:
        con = _conn()
        if con is None:
            has_db = False
            con = None

    if has_db and con:
        race_row = con.execute(
            "SELECT * FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()

        if race_row:
            race = dict(race_row)
            race["hd_display"] = _hd_display(race["hd"])

            bets_rows = con.execute(
                """
                SELECT strategy, combo, amount, odds_at_bet, ev_at_bet,
                       prob, status, payout
                FROM bets
                WHERE race_id = ?
                ORDER BY strategy, amount DESC
                """,
                (race_id,),
            ).fetchall()

            for b in bets_rows:
                strat = b["strategy"]
                if strat not in bets_by_strategy:
                    bets_by_strategy[strat] = []
                bets_by_strategy[strat].append(dict(b))

        con.close()

    # 戦略ごとの小計
    strategy_totals = {}
    for strat, blist in bets_by_strategy.items():
        total_bet    = sum(b["amount"] for b in blist)
        total_return = sum(b["payout"] or 0 for b in blist)
        strategy_totals[strat] = {
            "total_bet":    total_bet,
            "total_return": total_return,
            "profit":       total_return - total_bet,
            "n_bets":       len(blist),
        }
    for strat in _ordered_strategy_names(set(strategy_totals), include_configured=True):
        strategy_totals.setdefault(strat, {
            "total_bet": 0,
            "total_return": 0,
            "profit": 0,
            "n_bets": 0,
        })

    # 的中コンボの予想確率（いずれかの戦略でベットしていれば取得できる）
    result_prob: float | None = None
    if race and race.get("result_combo"):
        result_c = race["result_combo"]
        for blist in bets_by_strategy.values():
            for b in blist:
                if b["combo"] == result_c and b.get("prob") is not None:
                    result_prob = b["prob"]
                    break
            if result_prob is not None:
                break

    return render_template(
        "monitor/race_detail.html",
        race=race,
        race_id=race_id,
        bets_by_strategy=bets_by_strategy,
        strategy_totals=strategy_totals,
        result_prob=result_prob,
        has_db=has_db,
        profit_class=_profit_class,
    )


# -------------------------------------------------------
# 累積履歴  GET /monitor/history
# -------------------------------------------------------

@monitor_bp.route("/history")
def history():
    days = int(request.args.get("days", 7))
    has_db = DB_PATH.exists()
    daily_rows = []
    chart_labels = []
    strategy_names = []
    chart_roi = {}
    chart_cum = {}

    if has_db:
        con = _conn()
        rows = con.execute(
            """
            SELECT
                r.hd,
                b.strategy,
                COALESCE(SUM(b.amount), 0)                     AS total_bet,
                COALESCE(SUM(b.payout), 0)                     AS total_return,
                COUNT(CASE WHEN b.status='win' THEN 1 END)     AS hit_count,
                COUNT(DISTINCT r.race_id)                       AS races_bet
            FROM races r
            JOIN bets b ON b.race_id = r.race_id
            WHERE b.status != 'pending'
              AND r.hd >= ?
            GROUP BY r.hd, b.strategy
            ORDER BY r.hd
            """,
            ((date.today() - timedelta(days=days - 1)).strftime("%Y%m%d"),),
        ).fetchall()
        con.close()

        # hd × strategy に整理
        from collections import defaultdict
        day_map: dict = defaultdict(dict)
        all_hds = set()
        all_strategies = set()
        for row in rows:
            day_map[row["hd"]][row["strategy"]] = dict(row)
            all_hds.add(row["hd"])
            all_strategies.add(row["strategy"])

        preferred_order = [
            "kelly", "ip", "strict_flat", "true_kelly_cap",
            "dutch_value", "ip_conservative",
        ]
        all_strategies.update(STRATEGIES.keys())
        strategy_names = _ordered_strategy_names(all_strategies)

        for hd in sorted(all_hds):
            strategies = {}
            for strategy in strategy_names:
                row = day_map[hd].get(strategy, {})
                total_bet = row.get("total_bet", 0)
                total_return = row.get("total_return", 0)
                strategies[strategy] = {
                    "total_bet": total_bet,
                    "total_return": total_return,
                    "profit": total_return - total_bet,
                    "roi": round(total_return / total_bet * 100 if total_bet else 0, 1),
                    "hit": row.get("hit_count", 0),
                    "races_bet": row.get("races_bet", 0),
                }
            daily_rows.append({
                "hd": hd,
                "hd_display": _hd_display(hd),
                "strategies": strategies,
            })

        # Chart.js 用データ
        cumulative = {strategy: 0 for strategy in strategy_names}
        chart_roi = {strategy: [] for strategy in strategy_names}
        chart_cum = {strategy: [] for strategy in strategy_names}
        for d in daily_rows:
            chart_labels.append(d["hd_display"])
            for strategy in strategy_names:
                stats = d["strategies"][strategy]
                chart_roi[strategy].append(stats["roi"])
                cumulative[strategy] += stats["profit"]
                chart_cum[strategy].append(cumulative[strategy])

    return render_template(
        "monitor/history.html",
        days=days,
        daily_rows=daily_rows,
        strategy_names=strategy_names,
        chart_labels=chart_labels,
        chart_roi=chart_roi,
        chart_cum=chart_cum,
        has_db=has_db,
        profit_class=_profit_class,
    )

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
    strategies = strategy_summary(hd) if has_db else []

    # レース一覧（ステータス集計用）
    races = []
    if con := _conn():
        rows = con.execute(
            """
            SELECT r.race_id, r.venue, r.rno, r.stime,
                   r.result_combo, r.result_payout,
                   r.predicted_at,
                   COUNT(b.id) AS n_bets
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

    if has_db:
        con = _conn()
        rows = con.execute(
            """
            SELECT
                r.race_id, r.venue, r.rno, r.stime,
                r.predicted_at, r.result_combo, r.result_payout,
                -- 戦略別賭け金
                COALESCE(SUM(CASE WHEN b.strategy='kelly' THEN b.amount ELSE 0 END), 0) AS kelly_bet,
                COALESCE(SUM(CASE WHEN b.strategy='ip'    THEN b.amount ELSE 0 END), 0) AS ip_bet,
                -- 戦略別払戻
                COALESCE(SUM(CASE WHEN b.strategy='kelly' THEN b.payout ELSE 0 END), 0) AS kelly_return,
                COALESCE(SUM(CASE WHEN b.strategy='ip'    THEN b.payout ELSE 0 END), 0) AS ip_return,
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
        con.close()

        for r in rows:
            kelly_profit = r["kelly_return"] - r["kelly_bet"]
            ip_profit    = r["ip_return"]    - r["ip_bet"]
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

            rows_out.append({
                "race_id":       r["race_id"],
                "venue":         r["venue"],
                "rno":           r["rno"],
                "stime":         r["stime"],
                "status":        status,
                "status_cls":    status_cls,
                "result_combo":  r["result_combo"],
                "result_payout": r["result_payout"],
                "kelly_bet":     r["kelly_bet"],
                "ip_bet":        r["ip_bet"],
                "kelly_profit":  kelly_profit,
                "ip_profit":     ip_profit,
                "win_count":     r["win_count"],
            })

    return render_template(
        "monitor/races.html",
        hd=hd,
        hd_display=_hd_display(hd),
        races=rows_out,
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

    return render_template(
        "monitor/race_detail.html",
        race=race,
        race_id=race_id,
        bets_by_strategy=bets_by_strategy,
        strategy_totals=strategy_totals,
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
    chart_kelly_roi = []
    chart_ip_roi = []
    chart_kelly_cum = []
    chart_ip_cum = []

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
        for row in rows:
            day_map[row["hd"]][row["strategy"]] = dict(row)
            all_hds.add(row["hd"])

        for hd in sorted(all_hds):
            kelly = day_map[hd].get("kelly", {})
            ip    = day_map[hd].get("ip", {})

            kb = kelly.get("total_bet", 0)
            kr = kelly.get("total_return", 0)
            ib = ip.get("total_bet", 0)
            ir = ip.get("total_return", 0)

            daily_rows.append({
                "hd":          hd,
                "hd_display":  _hd_display(hd),
                "kelly_bet":   kb,
                "kelly_return":kr,
                "kelly_profit":kr - kb,
                "kelly_roi":   round(kr / kb * 100 if kb else 0, 1),
                "kelly_hit":   kelly.get("hit_count", 0),
                "ip_bet":      ib,
                "ip_return":   ir,
                "ip_profit":   ir - ib,
                "ip_roi":      round(ir / ib * 100 if ib else 0, 1),
                "ip_hit":      ip.get("hit_count", 0),
            })

        # Chart.js 用データ
        kelly_cum = ip_cum = 0
        for d in daily_rows:
            chart_labels.append(d["hd_display"])
            chart_kelly_roi.append(d["kelly_roi"])
            chart_ip_roi.append(d["ip_roi"])
            kelly_cum += d["kelly_profit"]
            ip_cum    += d["ip_profit"]
            chart_kelly_cum.append(kelly_cum)
            chart_ip_cum.append(ip_cum)

    return render_template(
        "monitor/history.html",
        days=days,
        daily_rows=daily_rows,
        chart_labels=chart_labels,
        chart_kelly_roi=chart_kelly_roi,
        chart_ip_roi=chart_ip_roi,
        chart_kelly_cum=chart_kelly_cum,
        chart_ip_cum=chart_ip_cum,
        has_db=has_db,
        profit_class=_profit_class,
    )

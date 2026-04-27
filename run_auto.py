"""
run_auto.py - dry-run auto monitoring entrypoint.

This script schedules predictions, collects results, and records dry-run profit
and loss. It never places real bets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def cmd_run(args: argparse.Namespace) -> None:
    from auto.orchestrator import run

    run(
        hd=args.date,
        budget=args.budget,
        min_edge=args.min_edge,
    )


def cmd_summary(args: argparse.Namespace) -> None:
    from auto.recorder import init_db, print_daily_summary

    init_db()
    print_daily_summary(args.date)


def cmd_show_db(args: argparse.Namespace) -> None:
    """Print race-level dry-run totals and dynamic strategy summaries."""
    import sqlite3

    from auto.recorder import DB_PATH, init_db, strategy_summary

    if not DB_PATH.exists():
        print("DB not found. Run `python run_auto.py` first.")
        return

    init_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    hd = args.date
    params: tuple[str, ...] = ()
    where = ""
    if hd:
        where = "WHERE r.hd = ?"
        params = (hd,)

    rows = con.execute(
        f"""
        SELECT
            r.race_id,
            r.venue,
            r.rno,
            r.stime,
            r.result_combo,
            r.result_payout,
            COUNT(DISTINCT b.id)             AS n_bets,
            COALESCE(SUM(b.amount), 0)       AS total_bet,
            COALESCE(SUM(b.payout), 0)       AS total_return,
            COALESCE(SUM(b.payout), 0)
              - COALESCE(SUM(b.amount), 0)   AS profit
        FROM races r
        LEFT JOIN bets b
          ON b.race_id = r.race_id
         AND b.status != 'pending'
        {where}
        GROUP BY r.race_id
        ORDER BY r.hd, r.stime, r.jcd, r.rno
        """,
        params,
    ).fetchall()

    if not rows:
        print("No races found.")
        con.close()
        return

    print(
        f"\n{'race':^22}  {'time':^5}  {'result':^8}  {'payout':^9}  "
        f"{'total_bet':^10}  {'return':^10}  {'profit':^10}"
    )
    print("-" * 88)

    total_bet = 0
    total_return = 0
    for row in rows:
        result = row["result_combo"] or "-"
        payout = f"{row['result_payout']:,}" if row["result_payout"] else "-"
        bet = f"{row['total_bet']:,}" if row["total_bet"] else "0"
        ret = f"{row['total_return']:,}" if row["total_return"] else "0"
        profit = f"{row['profit']:+,}"

        print(
            f"  {row['venue']:>6} {row['rno']:>2}R  {row['stime']}  "
            f"{result:>8}  {payout:>9}  {bet:>10}  {ret:>10}  {profit:>10}"
        )
        total_bet += row["total_bet"] or 0
        total_return += row["total_return"] or 0

    print("-" * 88)
    profit = total_return - total_bet
    roi = total_return / total_bet * 100 if total_bet else 0.0
    print(
        f"{'TOTAL':>42}  bet {total_bet:>10,}  "
        f"return {total_return:>10,}  profit {profit:>+10,}  ROI:{roi:.1f}%"
    )
    print()
    con.close()

    rows_s = strategy_summary(hd)
    if rows_s:
        print(
            f"  {'strategy':20s}  {'races':>6}  {'bet':>10}  "
            f"{'return':>10}  {'profit':>10}  {'ROI':>8}  {'hit':>8}"
        )
        print("  " + "-" * 82)
        for row in rows_s:
            print(
                f"  {row['strategy']:20s}  {row['races_bet']:>6}  "
                f"{row['total_bet']:>10,}  {row['total_return']:>10,}  "
                f"{row['profit']:>+10,}  {row['roi_pct']:>7.1f}%  "
                f"{row['hit_count']:>2}({row['hit_rate_pct']:.0f}%)"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boat race AI dry-run auto monitoring.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYYMMDD. Defaults to today.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="Dry-run budget per race in yen.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        dest="min_edge",
        help="Minimum raw EV threshold. 0.05 means 5%%.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the daily summary and exit.",
    )
    parser.add_argument(
        "--show-db",
        action="store_true",
        dest="show_db",
        help="Print dry-run database contents and exit.",
    )

    args = parser.parse_args()

    if args.summary:
        cmd_summary(args)
    elif args.show_db:
        cmd_show_db(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()

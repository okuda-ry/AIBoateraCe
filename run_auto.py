"""
run_auto.py — 競艇 AI 自動モニタリングの起動スクリプト。

【動作概要】
  1. 当日のレーススケジュールを取得
  2. 各レースの発走 5分前に予測を実行（ドライラン / 実投票なし）
  3. 発走 35分後に結果を取得して損益を計算
  4. data/auto.db に全記録を保存

【使い方】
  # 今日の全場を監視（ドライラン）
  python run_auto.py

  # 予算・最低EV を変更
  python run_auto.py --budget 2000 --min-edge 0.08

  # 特定の日付を指定（過去日のキャッチアップ確認など）
  python run_auto.py --date 20260405

  # 日次サマリーだけを表示して終了
  python run_auto.py --summary

  # データベースの内容を確認
  python run_auto.py --show-db

【注意】
  実際の投票は行いません。
  既存の Web アプリ (python app.py) とは独立して動作します。
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def cmd_run(args):
    from auto.orchestrator import run
    run(
        hd       = args.date,
        budget   = args.budget,
        min_edge = args.min_edge,
    )


def cmd_summary(args):
    from auto.recorder import init_db, print_daily_summary
    init_db()
    print_daily_summary(args.date)


def cmd_show_db(args):
    """DB の内容をテキスト形式で表示する（レース一覧 + 戦略別サマリー）。"""
    import sqlite3
    from auto.recorder import DB_PATH, init_db, strategy_summary

    if not DB_PATH.exists():
        print("DB が見つかりません。まず run_auto.py を実行してください。")
        return

    init_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    hd = args.date
    where = f"WHERE r.hd = '{hd}'" if hd else ""

    rows = con.execute(f"""
        SELECT
            r.race_id, r.venue, r.rno, r.stime,
            r.confidence,
            r.result_combo, r.result_payout,
            COUNT(DISTINCT b.id)                       AS n_bets,
            COALESCE(SUM(CASE WHEN b.strategy='kelly' THEN b.amount ELSE 0 END), 0) AS kelly_bet,
            COALESCE(SUM(CASE WHEN b.strategy='ip'    THEN b.amount ELSE 0 END), 0) AS ip_bet,
            COALESCE(SUM(CASE WHEN b.strategy='bayes' THEN b.amount ELSE 0 END), 0) AS bayes_bet,
            COALESCE(SUM(b.payout), 0)                 AS total_return,
            COALESCE(SUM(b.payout), 0)
                - COALESCE(SUM(b.amount), 0)           AS profit
        FROM races r
        LEFT JOIN bets b ON b.race_id = r.race_id AND b.status != 'pending'
        {where}
        GROUP BY r.race_id
        ORDER BY r.hd, r.stime, r.jcd, r.rno
    """).fetchall()

    if not rows:
        print("該当するレースが見つかりません。")
        con.close()
        return

    print(f"\n{'レース':^22}  {'時刻':^5}  {'結果':^8}  {'払戻':^7}  "
          f"{'kelly':^6}  {'ip':^6}  {'bayes':^6}  {'回収':^6}  {'損益':^7}")
    print("-" * 90)

    total_bet = total_return = 0
    for r in rows:
        result  = r["result_combo"] or "未確定"
        payout  = f"{r['result_payout']:,}" if r["result_payout"] else "  —"
        kelly   = f"{r['kelly_bet']:,}" if r["kelly_bet"] else "  0"
        ip      = f"{r['ip_bet']:,}"    if r["ip_bet"]    else "  0"
        bayes   = f"{r['bayes_bet']:,}" if r["bayes_bet"] else "  0"
        ret     = f"{r['total_return']:,}" if r["total_return"] else "  0"
        profit  = f"{r['profit']:+,}"
        print(f"  {r['venue']:>6} {r['rno']:>2}R  {r['stime']}  "
              f"{result:>8}  {payout:>7}  "
              f"{kelly:>6}  {ip:>6}  {bayes:>6}  "
              f"{ret:>6}  {profit:>7}")
        total_bet    += (r["kelly_bet"] or 0) + (r["ip_bet"] or 0) + (r["bayes_bet"] or 0)
        total_return += r["total_return"] or 0

    print("-" * 90)
    profit = total_return - total_bet
    roi    = total_return / total_bet * 100 if total_bet else 0
    print(f"  {'合計':>28}  "
          f"{'賭計':>10} {total_bet:>6,}  "
          f"{'回収':>2} {total_return:>6,}  {profit:>+7,}  ROI:{roi:.1f}%")
    print()
    con.close()

    # 戦略別サマリー
    rows_s = strategy_summary(hd)
    if rows_s:
        print(f"  {'戦略':8s}  {'賭レース':>6}  {'賭金':>8}  {'回収':>8}  {'損益':>8}  {'ROI':>7}  {'的中':>5}")
        print("  " + "-" * 62)
        for r in rows_s:
            print(
                f"  {r['strategy']:8s}  {r['races_bet']:>6}  "
                f"{r['total_bet']:>8,}  {r['total_return']:>8,}  "
                f"{r['profit']:>+8,}  {r['roi_pct']:>6.1f}%  "
                f"{r['hit_count']:>2}({r['hit_rate_pct']:.0f}%)"
            )
        print()


# -------------------------------------------------------
# エントリポイント
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="競艇 AI 自動モニタリング（ドライラン）",
    )
    parser.add_argument("--date",     type=str,   default=None,
                        help="対象日 YYYYMMDD（デフォルト: 今日）")
    parser.add_argument("--budget",   type=int,   default=1000,
                        help="1レースあたり仮想予算（円）")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        dest="min_edge",
                        help="最低期待値（デフォルト: 0.05 = 5%%）")
    parser.add_argument("--summary",  action="store_true",
                        help="日次サマリーを表示して終了")
    parser.add_argument("--show-db",  action="store_true",
                        dest="show_db",
                        help="DB の内容を表示して終了")

    args = parser.parse_args()

    if args.summary:
        cmd_summary(args)
    elif args.show_db:
        cmd_show_db(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()

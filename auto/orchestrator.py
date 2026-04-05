"""
auto/orchestrator.py — レーススケジュールに合わせて予測・結果収集を自動実行する。

動作モード: ドライラン（実際の投票は行わない）
  - 発走 PREDICT_BEFORE 分前 → 予測を実行し DB に記録
  - 発走 RESULT_AFTER 分後  → 結果を取得し損益を計算

既存の Web アプリ (app.py) と predict.py はそのまま使える状態を維持する。
"""

import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

# プロジェクトルートを sys.path に追加（auto/ から import できるよう）
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.date import DateTrigger

from data.scraper import scrape_schedule, scrape_result
from auto.recorder import (
    init_db, upsert_race, save_prediction,
    update_result, get_pending_result_races,
    print_daily_summary,
)

# -------------------------------------------------------
# 設定
# -------------------------------------------------------

PREDICT_BEFORE = 10    # 発走の何分前に予測を実行するか
RESULT_AFTER   = 35    # 発走の何分後に結果を取得するか
DEFAULT_BUDGET = 1000  # 1レースあたり仮想予算（円）
MIN_EDGE       = 0.05  # 最低期待値


# -------------------------------------------------------
# ジョブ関数
# -------------------------------------------------------

def job_predict(race: dict, budget: int, min_edge: float) -> None:
    """
    予測ジョブ: run_prediction() を呼び出して結果を DB に保存する。

    run_prediction は app.py に定義されており、Flask を起動せずに
    import して使用する。
    """
    race_id = race["race_id"]
    print(f"\n[orchestrator] 予測開始: {race['venue']} {race['rno']}R  ({race_id})")

    try:
        # app.py の run_prediction をそのまま利用
        from app import run_prediction
        result = run_prediction(
            jcd        = race["jcd"],
            hd         = race["hd"],
            rno        = race["rno"],
            venue_name = race["venue"],
            nichiji    = 1,
            budget     = budget,
            use_odds   = True,
            min_edge   = min_edge,
        )
        save_prediction(race_id, result)

        bets = result.get("bets", [])
        if bets:
            total = result.get("total_bet", 0)
            print(f"[orchestrator] ベット候補: {len(bets)}件 / {total:,}円 "
                  f"(ドライラン: 実際の投票は行いません)")
        else:
            print(f"[orchestrator] 見送り推奨（期待値プラスの組み合わせなし）")

    except Exception:
        print(f"[orchestrator] 予測エラー: {race_id}")
        traceback.print_exc()


def job_collect_result(race: dict) -> None:
    """
    結果収集ジョブ: レース結果を取得して損益を更新する。
    """
    race_id = race["race_id"]
    print(f"\n[orchestrator] 結果収集: {race['venue']} {race['rno']}R  ({race_id})")

    try:
        res = scrape_result(race["jcd"], race["hd"], race["rno"])
        if res["finished"]:
            update_result(race_id, res["combo"], res["payout"])
        else:
            print(f"[orchestrator] 結果未確定 — 5分後に再試行します")
            # 再試行は run() が pending_results を監視することで対応
    except Exception:
        print(f"[orchestrator] 結果取得エラー: {race_id}")
        traceback.print_exc()


# -------------------------------------------------------
# メイン実行
# -------------------------------------------------------

def run(hd: str = None, budget: int = DEFAULT_BUDGET,
        min_edge: float = MIN_EDGE) -> None:
    """
    当日のスケジュールを取得し、APScheduler でジョブを登録して実行する。

    Parameters
    ----------
    hd       : 対象日 YYYYMMDD。None = 今日。
    budget   : 1レースあたりの仮想予算（円）
    min_edge : 最低期待値（バリューベット閾値）
    """
    if hd is None:
        hd = date.today().strftime("%Y%m%d")

    print("=" * 60)
    print(f"  競艇 AI 自動モニタリング  [{hd}]  (ドライラン)")
    print(f"  予算: {budget:,}円/レース  最低EV: {min_edge*100:.0f}%")
    print("=" * 60)

    init_db()

    # スケジュール取得
    races = scrape_schedule(hd)
    if not races:
        print("[orchestrator] 本日のレースが見つかりませんでした。終了します。")
        return

    # レースを DB に登録
    for race in races:
        upsert_race(race)

    now = datetime.now()
    scheduler = BlockingScheduler(timezone="Asia/Tokyo")
    scheduled_count = 0

    for race in races:
        stime_str = race["stime"]   # "HH:MM"
        try:
            stime_dt = datetime.strptime(f"{hd} {stime_str}", "%Y%m%d %H:%M")
        except ValueError:
            print(f"[orchestrator] 時刻パース失敗: {race['race_id']} stime={stime_str}")
            continue

        predict_at = stime_dt - timedelta(minutes=PREDICT_BEFORE)
        result_at  = stime_dt + timedelta(minutes=RESULT_AFTER)

        # 既に過ぎた予測時刻は即時実行（起動時のキャッチアップ）
        if predict_at <= now:
            if stime_dt > now:
                # レースがまだ始まっていなければ今すぐ予測
                predict_at = now + timedelta(seconds=5 + races.index(race) * 3)
            else:
                # レース自体も過ぎている → 結果取得のみ
                predict_at = None

        if predict_at:
            scheduler.add_job(
                func     = job_predict,
                trigger  = DateTrigger(run_date=predict_at),
                kwargs   = {"race": race, "budget": budget, "min_edge": min_edge},
                id       = f"predict_{race['race_id']}",
                name     = f"予測 {race['venue']} {race['rno']}R",
                misfire_grace_time = 120,
            )
            scheduled_count += 1

        # 結果収集は未来のレースのみスケジュール
        if result_at > now:
            scheduler.add_job(
                func     = job_collect_result,
                trigger  = DateTrigger(run_date=result_at),
                kwargs   = {"race": race},
                id       = f"result_{race['race_id']}",
                name     = f"結果 {race['venue']} {race['rno']}R",
                misfire_grace_time = 300,
            )

    print(f"\n[orchestrator] ジョブ登録: 予測 {scheduled_count} 件")
    print("[orchestrator] スケジューラ起動中... (Ctrl+C で停止)\n")

    # 日次サマリーを毎時 59分に表示
    scheduler.add_job(
        func    = print_daily_summary,
        trigger = "cron",
        minute  = 59,
        kwargs  = {"hd": hd},
        id      = "hourly_summary",
        name    = "毎時サマリー",
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[orchestrator] 停止しました。")
    finally:
        print_daily_summary(hd)

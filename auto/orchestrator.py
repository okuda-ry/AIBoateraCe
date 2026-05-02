"""
auto/orchestrator.py — レーススケジュールに合わせて予測・結果収集を自動実行する。

動作モード: ドライラン（実際の投票は行わない）
  - 発走 PREDICT_BEFORE 分前 → 予測を実行し DB に記録
  - 発走 RESULT_AFTER 分後  → 結果を取得し損益を計算
  - 毎日 DAILY_REPORT_TIME  → LINE で日次損益レポートを送信

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
    save_all_strategies,
    update_result, print_daily_summary, daily_summary,
)
from auto.notifier import notify_daily_summary, notify_discord_prediction

# -------------------------------------------------------
# 設定
# -------------------------------------------------------

PREDICT_BEFORE    = 5      # 発走の何分前に予測を実行するか
RESULT_AFTER      = 35     # 発走の何分後に結果を取得するか
DAILY_REPORT_HOUR = 21     # 日次 LINE レポートの送信時刻（時）
DAILY_REPORT_MIN  = 0
DEFAULT_BUDGET    = 1000   # 1レースあたり仮想予算（円）
MIN_EDGE          = 0.05   # 最低期待値


# -------------------------------------------------------
# ジョブ関数
# -------------------------------------------------------

def job_predict(race: dict, budget: int, min_edge: float) -> None:
    """
    予測ジョブ: run_prediction() を呼び出し、全戦略を実行して DB に保存する。
    run_prediction は app.py に定義されており、Flask を起動せずに利用する。
    """
    race_id = race["race_id"]
    print(f"\n[orchestrator] 予測開始: {race['venue']} {race['rno']}R  ({race_id})")

    try:
        from app import run_prediction
        from models.strategies import run_all_strategies, STRATEGIES

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

        probs_120 = result.get("_probs_120")
        odds_dict = result.get("_odds_dict", {})

        if probs_120 is None or not odds_dict:
            # オッズなし → Kelly のみ保存（旧互換）
            save_prediction(race_id, result)
            print(f"[orchestrator] オッズなし → Kelly のみ記録")
            return

        # ベイズ戦略: probs_120 を DB に保存するまでは無効
        # get_history_for_bayes() が返す probs=None のままだと
        # allocate_kelly 内の np.argsort(None) でクラッシュするため。
        # TODO: races テーブルに probs BLOB カラムを追加して有効化する
        strategies = dict(STRATEGIES)  # kelly + ip のみ

        # 全戦略を実行
        print(f"[orchestrator] 戦略実行中: {list(strategies.keys())}")
        strategy_bets = run_all_strategies(
            probs_120, odds_dict, budget=budget,
            strategies=strategies,
            min_edge=min_edge,
        )

        # DB に保存
        save_all_strategies(
            race_id       = race_id,
            strategy_bets = strategy_bets,
            odds_dict     = odds_dict,
            probs_120     = probs_120,
            confidence    = result.get("confidence", 0.0) / 100,
            has_odds      = bool(odds_dict),
        )

        any_bets = any(v for v in strategy_bets.values())
        if any_bets:
            print(f"[orchestrator] 全戦略完了 (ドライラン: 実際の投票は行いません)")
        else:
            print(f"[orchestrator] 全戦略: 見送り推奨（期待値プラスの組み合わせなし）")

        notify_discord_prediction(
            race          = race,
            strategy_bets = strategy_bets,
            odds_dict     = odds_dict,
            probs_120     = probs_120,
            confidence    = result.get("confidence", 0.0) / 100,
            budget        = budget,
            min_edge      = min_edge,
            prediction_result = result,
        )

    except Exception:
        print(f"[orchestrator] 予測エラー: {race_id}")
        traceback.print_exc()


def job_collect_result(race: dict) -> None:
    """結果収集ジョブ: レース結果を取得して損益を更新する。"""
    race_id = race["race_id"]
    print(f"\n[orchestrator] 結果収集: {race['venue']} {race['rno']}R  ({race_id})")

    try:
        res = scrape_result(race["jcd"], race["hd"], race["rno"])
        if res["finished"]:
            update_result(race_id, res["combo"], res["payout"])
        else:
            print(f"[orchestrator] 結果未確定（レース未完了の可能性）")
    except Exception:
        print(f"[orchestrator] 結果取得エラー: {race_id}")
        traceback.print_exc()


def job_line_daily_report(hd: str) -> None:
    """日次 LINE レポートジョブ: サマリーを集計して LINE に送信する。"""
    print(f"\n[orchestrator] 日次 LINE レポート送信: {hd}")
    print_daily_summary(hd)
    try:
        s = daily_summary(hd)
        notify_daily_summary(s)
    except Exception:
        print("[orchestrator] LINE 通知エラー")
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
    print(f"  予測タイミング: 発走 {PREDICT_BEFORE} 分前")
    print(f"  日次レポート:   {DAILY_REPORT_HOUR:02d}:{DAILY_REPORT_MIN:02d} (LINE)")
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
    predict_count = 0

    for race in races:
        stime_str = race["stime"]
        try:
            stime_dt = datetime.strptime(f"{hd} {stime_str}", "%Y%m%d %H:%M")
        except ValueError:
            print(f"[orchestrator] 時刻パース失敗: {race['race_id']} stime={stime_str}")
            continue

        predict_at = stime_dt - timedelta(minutes=PREDICT_BEFORE)
        result_at  = stime_dt + timedelta(minutes=RESULT_AFTER)

        # 予測時刻が過ぎていてもレースが始まっていなければ今すぐ予測
        if predict_at <= now:
            if stime_dt > now:
                predict_at = now + timedelta(seconds=5 + predict_count * 3)
            else:
                predict_at = None   # レース自体も過去 → 結果取得のみ

        if predict_at:
            scheduler.add_job(
                func               = job_predict,
                trigger            = DateTrigger(run_date=predict_at),
                kwargs             = {"race": race, "budget": budget, "min_edge": min_edge},
                id                 = f"predict_{race['race_id']}",
                name               = f"予測 {race['venue']} {race['rno']}R",
                misfire_grace_time = 120,
            )
            predict_count += 1

        if result_at > now:
            scheduler.add_job(
                func               = job_collect_result,
                trigger            = DateTrigger(run_date=result_at),
                kwargs             = {"race": race},
                id                 = f"result_{race['race_id']}",
                name               = f"結果 {race['venue']} {race['rno']}R",
                misfire_grace_time = 300,
            )

    # ── 日次 LINE レポート (DAILY_REPORT_HOUR:DAILY_REPORT_MIN) ──
    report_dt = datetime.strptime(
        f"{hd} {DAILY_REPORT_HOUR:02d}:{DAILY_REPORT_MIN:02d}", "%Y%m%d %H:%M"
    )
    if report_dt > now:
        scheduler.add_job(
            func               = job_line_daily_report,
            trigger            = DateTrigger(run_date=report_dt),
            kwargs             = {"hd": hd},
            id                 = "daily_line_report",
            name               = f"LINE日次レポート {DAILY_REPORT_HOUR:02d}:{DAILY_REPORT_MIN:02d}",
            misfire_grace_time = 600,
        )
        print(f"[orchestrator] LINE レポート: {report_dt.strftime('%H:%M')} に送信予定")
    else:
        # 既に報告時刻を過ぎている場合は即時送信
        job_line_daily_report(hd)

    print(f"[orchestrator] ジョブ登録: 予測 {predict_count} 件")
    print("[orchestrator] スケジューラ起動中... (Ctrl+C で停止)\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[orchestrator] 停止しました。")
    finally:
        # 終了時に必ずサマリーを表示
        print_daily_summary(hd)

"""
auto/recorder.py — 予測・ベット・結果を SQLite に記録する。

DB ファイル: data/auto.db  (git 管理外)
"""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "auto.db"


# -------------------------------------------------------
# 接続ユーティリティ
# -------------------------------------------------------

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# -------------------------------------------------------
# テーブル初期化
# -------------------------------------------------------

def init_db() -> None:
    """DB とテーブルを初期化する（既存テーブルは変更しない）。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS races (
                race_id          TEXT PRIMARY KEY,
                jcd              TEXT NOT NULL,
                venue            TEXT NOT NULL,
                hd               TEXT NOT NULL,
                rno              INTEGER NOT NULL,
                stime            TEXT NOT NULL,
                predicted_at     DATETIME,
                confidence       REAL,
                has_odds         INTEGER DEFAULT 0,
                result_combo     TEXT,
                result_payout    INTEGER,
                result_fetched_at DATETIME
            );

            CREATE TABLE IF NOT EXISTS bets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id      TEXT NOT NULL,
                combo        TEXT NOT NULL,
                amount       INTEGER NOT NULL,
                odds_at_bet  REAL,
                ev_at_bet    REAL,
                prob         REAL,
                status       TEXT DEFAULT 'pending',
                payout       INTEGER DEFAULT 0,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );
        """)
    print(f"[recorder] DB 初期化完了: {DB_PATH}")


# -------------------------------------------------------
# レース登録
# -------------------------------------------------------

def upsert_race(race: dict) -> None:
    """レーススケジュールを登録（重複時は無視）。"""
    with _conn() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO races (race_id, jcd, venue, hd, rno, stime)
            VALUES (:race_id, :jcd, :venue, :hd, :rno, :stime)
            """,
            race,
        )


# -------------------------------------------------------
# 予測結果の保存
# -------------------------------------------------------

def save_prediction(race_id: str, result: dict) -> None:
    """
    app.run_prediction() の戻り値から予測情報を保存する。

    Parameters
    ----------
    race_id : str
    result  : run_prediction() が返す dict
    """
    bets_list = result.get("bets", [])
    odds_dict = {b["combo"]: b.get("odds") for b in bets_list if b.get("odds")}
    has_odds  = bool(result.get("has_odds", False))

    with _conn() as con:
        con.execute(
            """
            UPDATE races
            SET predicted_at = :now, confidence = :conf, has_odds = :ho
            WHERE race_id = :rid
            """,
            {
                "now":  datetime.now(),
                "conf": result.get("confidence", 0.0),
                "ho":   int(has_odds),
                "rid":  race_id,
            },
        )

        # 既存の pending ベットを削除してから再登録
        con.execute(
            "DELETE FROM bets WHERE race_id = ? AND status = 'pending'",
            (race_id,),
        )

        for bet in bets_list:
            combo = bet["combo"]
            con.execute(
                """
                INSERT INTO bets (race_id, combo, amount, odds_at_bet, ev_at_bet, prob)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    combo,
                    bet.get("amount", 0),
                    bet.get("odds"),
                    bet.get("ev_pct", None) and bet["ev_pct"] / 100.0,
                    None,   # prob は result dict に含まれていれば追加可
                ),
            )

    n = len(bets_list)
    total = result.get("total_bet", 0)
    print(f"[recorder] 予測保存: {race_id}  ベット{n}件 / 合計{total:,}円")


# -------------------------------------------------------
# レース結果の更新
# -------------------------------------------------------

def update_result(race_id: str, combo: str, payout: int) -> dict:
    """
    結果を反映し、各ベットの損益を更新する。

    Returns
    -------
    summary : {"race_id", "combo", "payout", "total_bet", "total_return", "profit"}
    """
    with _conn() as con:
        con.execute(
            """
            UPDATE races
            SET result_combo = ?, result_payout = ?, result_fetched_at = ?
            WHERE race_id = ?
            """,
            (combo, payout, datetime.now(), race_id),
        )

        bets = con.execute(
            "SELECT id, combo, amount FROM bets WHERE race_id = ? AND status = 'pending'",
            (race_id,),
        ).fetchall()

        total_bet    = 0
        total_return = 0
        for bet in bets:
            amount  = bet["amount"]
            total_bet += amount
            if bet["combo"] == combo and payout > 0:
                # 払戻 = (賭け金 / 100) × 払戻金
                ret = int(amount / 100 * payout)
                total_return += ret
                con.execute(
                    "UPDATE bets SET status = 'win', payout = ? WHERE id = ?",
                    (ret, bet["id"]),
                )
            else:
                con.execute(
                    "UPDATE bets SET status = 'lose', payout = 0 WHERE id = ?",
                    (bet["id"],),
                )

    profit = total_return - total_bet
    print(
        f"[recorder] 結果更新: {race_id}  {combo}  "
        f"払戻{payout:,}円  "
        f"賭:{total_bet:,}円  "
        f"回収:{total_return:,}円  "
        f"損益:{profit:+,}円"
    )
    return {
        "race_id":      race_id,
        "combo":        combo,
        "payout":       payout,
        "total_bet":    total_bet,
        "total_return": total_return,
        "profit":       profit,
    }


# -------------------------------------------------------
# 結果未確定のレース一覧
# -------------------------------------------------------

def get_pending_result_races(hd: str = None) -> list[dict]:
    """予測済みかつ結果未取得のレースを返す。"""
    if hd is None:
        hd = date.today().strftime("%Y%m%d")
    with _conn() as con:
        rows = con.execute(
            """
            SELECT race_id, jcd, hd, rno, stime
            FROM races
            WHERE hd = ? AND predicted_at IS NOT NULL AND result_combo IS NULL
            ORDER BY stime, jcd, rno
            """,
            (hd,),
        ).fetchall()
    return [dict(r) for r in rows]


# -------------------------------------------------------
# 日次サマリー
# -------------------------------------------------------

def daily_summary(hd: str = None) -> dict:
    """
    指定日の損益サマリーを返す。

    Returns
    -------
    {
        "date", "races_scheduled", "races_predicted", "races_bet",
        "races_skip", "total_bet", "total_return", "profit", "roi_pct",
        "hit_count", "hit_rate_pct"
    }
    """
    if hd is None:
        hd = date.today().strftime("%Y%m%d")

    with _conn() as con:
        races_scheduled = con.execute(
            "SELECT COUNT(*) FROM races WHERE hd = ?", (hd,)
        ).fetchone()[0]

        races_predicted = con.execute(
            "SELECT COUNT(*) FROM races WHERE hd = ? AND predicted_at IS NOT NULL",
            (hd,),
        ).fetchone()[0]

        agg = con.execute(
            """
            SELECT
                COUNT(DISTINCT b.race_id)          AS races_bet,
                COALESCE(SUM(b.amount), 0)         AS total_bet,
                COALESCE(SUM(b.payout), 0)         AS total_return,
                COUNT(CASE WHEN b.status='win' THEN 1 END) AS hit_count
            FROM bets b
            JOIN races r ON r.race_id = b.race_id
            WHERE r.hd = ? AND b.status != 'pending'
            """,
            (hd,),
        ).fetchone()

    races_bet    = agg["races_bet"]
    total_bet    = agg["total_bet"]
    total_return = agg["total_return"]
    hit_count    = agg["hit_count"]
    profit       = total_return - total_bet
    roi_pct      = total_return / total_bet * 100 if total_bet > 0 else 0.0
    hit_rate     = hit_count / races_bet * 100 if races_bet > 0 else 0.0

    return {
        "date":            hd,
        "races_scheduled": races_scheduled,
        "races_predicted": races_predicted,
        "races_bet":       races_bet,
        "races_skip":      races_predicted - races_bet,
        "total_bet":       total_bet,
        "total_return":    total_return,
        "profit":          profit,
        "roi_pct":         round(roi_pct, 2),
        "hit_count":       hit_count,
        "hit_rate_pct":    round(hit_rate, 1),
    }


def print_daily_summary(hd: str = None) -> None:
    """日次サマリーをコンソールに表示する。"""
    s = daily_summary(hd)
    print()
    print("=" * 52)
    print(f"  日次サマリー  {s['date']}")
    print("=" * 52)
    print(f"  スケジュール済み : {s['races_scheduled']:>4} レース")
    print(f"  予測実行済み     : {s['races_predicted']:>4} レース")
    print(f"  ベットあり       : {s['races_bet']:>4} レース")
    print(f"  見送り           : {s['races_skip']:>4} レース")
    print(f"  的中             : {s['hit_count']:>4} 回  ({s['hit_rate_pct']:.1f}%)")
    print(f"  総賭け金         : {s['total_bet']:>8,} 円")
    print(f"  総払戻           : {s['total_return']:>8,} 円")
    print(f"  損益             : {s['profit']:>+8,} 円")
    print(f"  回収率           : {s['roi_pct']:>7.2f} %")
    print("=" * 52)
    print()

"""
auto/recorder.py — 予測・ベット・結果を SQLite に記録する。

DB ファイル: data/auto.db  (git 管理外)

複数戦略の比較のため bets テーブルに strategy カラムを持つ。
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
    """DB とテーブルを初期化する。既存テーブルへのカラム追加も行う。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS races (
                race_id           TEXT PRIMARY KEY,
                jcd               TEXT NOT NULL,
                venue             TEXT NOT NULL,
                hd                TEXT NOT NULL,
                rno               INTEGER NOT NULL,
                stime             TEXT NOT NULL,
                predicted_at      DATETIME,
                confidence        REAL,
                has_odds          INTEGER DEFAULT 0,
                result_combo      TEXT,
                result_payout     INTEGER,
                result_fetched_at DATETIME
            );

            CREATE TABLE IF NOT EXISTS bets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id      TEXT NOT NULL,
                strategy     TEXT NOT NULL DEFAULT 'kelly',
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

        # 既存 DB に strategy カラムがなければ追加（マイグレーション）
        cols = [r[1] for r in con.execute("PRAGMA table_info(bets)").fetchall()]
        if "strategy" not in cols:
            con.execute("ALTER TABLE bets ADD COLUMN strategy TEXT NOT NULL DEFAULT 'kelly'")
            print("[recorder] bets テーブルに strategy カラムを追加しました")

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
# 複数戦略のベット保存
# -------------------------------------------------------

def save_strategy_bets(
    race_id:   str,
    strategy:  str,
    bets_dict: dict,
    odds_dict: dict,
    probs_120: "np.ndarray | None" = None,
) -> None:
    """
    1戦略分のベットを DB に保存する。

    Parameters
    ----------
    race_id   : レース ID
    strategy  : 戦略名 ("kelly" / "ip" / "bayes" など)
    bets_dict : {combo_str: amount}
    odds_dict : {combo_str: float}
    probs_120 : (120,) 予測確率配列（省略可）
    """
    import numpy as np
    from models.strategies import COMBO_IDX

    with _conn() as con:
        # 既存の pending ベット（同戦略）を削除してから再登録
        con.execute(
            "DELETE FROM bets WHERE race_id = ? AND strategy = ? AND status = 'pending'",
            (race_id, strategy),
        )

        for combo, amount in bets_dict.items():
            odds = odds_dict.get(combo)
            ev   = None
            prob = None
            if odds and probs_120 is not None:
                idx  = COMBO_IDX.get(combo)
                if idx is not None:
                    prob = float(probs_120[idx])
                    ev   = prob * odds - 1.0
            con.execute(
                """
                INSERT INTO bets
                  (race_id, strategy, combo, amount, odds_at_bet, ev_at_bet, prob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (race_id, strategy, combo, amount, odds, ev, prob),
            )

    total = sum(bets_dict.values())
    n     = len(bets_dict)
    print(f"[recorder] {strategy:8s}: {n}通り / {total:,}円  ({race_id})")


def save_all_strategies(
    race_id:          str,
    strategy_bets:    dict,   # {strategy_name: bets_dict}
    odds_dict:        dict,
    probs_120:        "np.ndarray | None" = None,
    confidence:       float = 0.0,
    has_odds:         bool  = False,
) -> None:
    """
    全戦略のベット結果をまとめて保存し、races テーブルも更新する。
    """
    with _conn() as con:
        con.execute(
            """
            UPDATE races
            SET predicted_at = ?, confidence = ?, has_odds = ?
            WHERE race_id = ?
            """,
            (datetime.now(), confidence, int(has_odds), race_id),
        )

    for strategy, bets_dict in strategy_bets.items():
        save_strategy_bets(race_id, strategy, bets_dict, odds_dict, probs_120)


# -------------------------------------------------------
# 後方互換: 旧 save_prediction（Web アプリから呼ばれる）
# -------------------------------------------------------

def save_prediction(race_id: str, result: dict) -> None:
    """app.run_prediction() の戻り値から kelly 戦略として保存する。"""
    bets_list = result.get("bets", [])
    has_odds  = bool(result.get("has_odds", False))
    odds_dict = {b["combo"]: b.get("odds") for b in bets_list if b.get("odds")}
    bets_dict = {b["combo"]: b.get("amount", 0) for b in bets_list}

    with _conn() as con:
        con.execute(
            """
            UPDATE races
            SET predicted_at = ?, confidence = ?, has_odds = ?
            WHERE race_id = ?
            """,
            (datetime.now(), result.get("confidence", 0.0), int(has_odds), race_id),
        )

    save_strategy_bets(race_id, "kelly", bets_dict, odds_dict)

    total = result.get("total_bet", 0)
    print(f"[recorder] 予測保存(kelly): {race_id}  {len(bets_dict)}件 / {total:,}円")


# -------------------------------------------------------
# レース結果の更新（全戦略まとめて）
# -------------------------------------------------------

def update_result(race_id: str, combo: str, payout: int) -> dict:
    """結果を反映し、全戦略のベット損益を更新する。"""
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
            "SELECT id, combo, amount, strategy FROM bets WHERE race_id = ? AND status = 'pending'",
            (race_id,),
        ).fetchall()

        total_bet = total_return = 0
        for bet in bets:
            amount = bet["amount"]
            total_bet += amount
            if bet["combo"] == combo and payout > 0:
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
        f"払戻{payout:,}円  賭:{total_bet:,}円  "
        f"回収:{total_return:,}円  損益:{profit:+,}円"
    )
    return {
        "race_id": race_id, "combo": combo, "payout": payout,
        "total_bet": total_bet, "total_return": total_return, "profit": profit,
    }


# -------------------------------------------------------
# 結果未確定のレース一覧
# -------------------------------------------------------

def get_pending_result_races(hd: str = None) -> list[dict]:
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
# 日次サマリー（全体）
# -------------------------------------------------------

def daily_summary(hd: str = None) -> dict:
    if hd is None:
        hd = date.today().strftime("%Y%m%d")

    with _conn() as con:
        races_scheduled = con.execute(
            "SELECT COUNT(*) FROM races WHERE hd = ?", (hd,)
        ).fetchone()[0]

        races_predicted = con.execute(
            "SELECT COUNT(*) FROM races WHERE hd = ? AND predicted_at IS NOT NULL", (hd,)
        ).fetchone()[0]

        agg = con.execute(
            """
            SELECT
                COUNT(DISTINCT b.race_id)                      AS races_bet,
                COALESCE(SUM(b.amount), 0)                     AS total_bet,
                COALESCE(SUM(b.payout), 0)                     AS total_return,
                COUNT(CASE WHEN b.status = 'win' THEN 1 END)   AS hit_count
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


# -------------------------------------------------------
# 戦略別サマリー
# -------------------------------------------------------

def strategy_summary(hd: str = None) -> list[dict]:
    """
    戦略ごとの損益サマリーを返す。

    Returns
    -------
    [
        {"strategy": "kelly", "races_bet": 5, "total_bet": 5000,
         "total_return": 4500, "profit": -500, "roi_pct": 90.0,
         "hit_count": 0, "hit_rate_pct": 0.0},
        ...
    ]
    """
    if hd is None:
        hd = date.today().strftime("%Y%m%d")

    with _conn() as con:
        rows = con.execute(
            """
            SELECT
                b.strategy,
                COUNT(DISTINCT b.race_id)                      AS races_bet,
                COALESCE(SUM(b.amount), 0)                     AS total_bet,
                COALESCE(SUM(b.payout), 0)                     AS total_return,
                COUNT(CASE WHEN b.status = 'win' THEN 1 END)   AS hit_count
            FROM bets b
            JOIN races r ON r.race_id = b.race_id
            WHERE r.hd = ? AND b.status != 'pending'
            GROUP BY b.strategy
            ORDER BY b.strategy
            """,
            (hd,),
        ).fetchall()

    result = []
    for row in rows:
        tb  = row["total_bet"]
        tr  = row["total_return"]
        hit = row["hit_count"]
        rb  = row["races_bet"]
        result.append({
            "strategy":     row["strategy"],
            "races_bet":    rb,
            "total_bet":    tb,
            "total_return": tr,
            "profit":       tr - tb,
            "roi_pct":      round(tr / tb * 100 if tb > 0 else 0.0, 2),
            "hit_count":    hit,
            "hit_rate_pct": round(hit / rb * 100 if rb > 0 else 0.0, 1),
        })
    return result


# -------------------------------------------------------
# 表示ユーティリティ
# -------------------------------------------------------

def print_daily_summary(hd: str = None) -> None:
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
    print()

    # 戦略別内訳
    rows = strategy_summary(hd)
    if rows:
        print(f"  {'戦略':8s}  {'賭':>4}  {'賭金':>7}  {'回収':>7}  {'損益':>7}  {'ROI':>7}  {'的中':>4}")
        print("  " + "-" * 54)
        for r in rows:
            print(
                f"  {r['strategy']:8s}  {r['races_bet']:>4}  "
                f"{r['total_bet']:>7,}  {r['total_return']:>7,}  "
                f"{r['profit']:>+7,}  {r['roi_pct']:>6.1f}%  "
                f"{r['hit_count']:>2}({r['hit_rate_pct']:.0f}%)"
            )
    print("=" * 52)
    print()


def get_history_for_bayes(hd_from: str = None, hd_to: str = None) -> list[dict]:
    """
    ベイズ最適化の学習用に過去レースの予測・結果データを返す。

    Returns
    -------
    [{"probs": None, "odds_dict": {}, "result_combo": str, "result_payout": int}, ...]
    注意: probs は DB に保存していないため None。odds は ev_at_bet から逆算不可。
          将来的に probs を保存するカラムを追加すれば完全に使える。
    """
    with _conn() as con:
        rows = con.execute(
            """
            SELECT
                r.race_id, r.result_combo, r.result_payout,
                b.combo, b.amount, b.odds_at_bet, b.ev_at_bet, b.strategy
            FROM races r
            JOIN bets b ON b.race_id = r.race_id
            WHERE r.result_combo IS NOT NULL
              AND b.strategy = 'kelly'
              AND (:from IS NULL OR r.hd >= :from)
              AND (:to   IS NULL OR r.hd <= :to)
            ORDER BY r.hd, r.rno
            """,
            {"from": hd_from, "to": hd_to},
        ).fetchall()

    # race_id ごとにまとめる
    from collections import defaultdict
    race_map: dict = defaultdict(lambda: {"odds_dict": {}, "result_combo": "", "result_payout": 0})
    for row in rows:
        rid = row["race_id"]
        race_map[rid]["result_combo"]  = row["result_combo"]
        race_map[rid]["result_payout"] = row["result_payout"]
        if row["odds_at_bet"]:
            race_map[rid]["odds_dict"][row["combo"]] = row["odds_at_bet"]

    return [
        {"probs": None, **v}
        for v in race_map.values()
        if v["result_combo"]
    ]

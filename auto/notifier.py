"""
auto/notifier.py — LINE Messaging API による通知モジュール。

設定方法:
  プロジェクトルートに .env ファイルを作成し、以下を記載:

    LINE_CHANNEL_ACCESS_TOKEN=<チャネルアクセストークン（長期）>
    LINE_USER_ID=<自分のユーザーID (Uxxxxxxxx...)>

  ※ Autrader プロジェクトの .env をそのままコピーして使用できます。
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------
# .env 読み込み
# -------------------------------------------------------

_ENV_PATH = Path(__file__).parent.parent / ".env"


def _load_env() -> None:
    if not _ENV_PATH.exists():
        return
    with open(_ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()


def _get_config() -> tuple[str, str]:
    token   = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")
    if not token or not user_id:
        raise ValueError(
            "LINE_CHANNEL_ACCESS_TOKEN と LINE_USER_ID が未設定です。\n"
            f"{_ENV_PATH} に記載してください。"
        )
    return token, user_id


# -------------------------------------------------------
# 送信コア
# -------------------------------------------------------

def send_line_message(text: str) -> bool:
    """
    LINE にテキストメッセージを送信する。

    Returns
    -------
    True = 成功 / False = 失敗（設定未完了を含む）
    """
    try:
        token, user_id = _get_config()
    except ValueError as e:
        print(f"[LINE] 設定エラー: {e}")
        return False

    url  = "https://api.line.me/v2/bot/message/push"
    body = json.dumps({
        "to":       user_id,
        "messages": [{"type": "text", "text": text}],
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data    = body,
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {token}",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
            print(f"[LINE] HTTP {resp.status}")
            return False
    except urllib.error.HTTPError as e:
        print(f"[LINE] HTTPError {e.code}: {e.read().decode()}")
        return False
    except Exception as e:
        print(f"[LINE] 送信エラー: {e}")
        return False


# -------------------------------------------------------
# 競艇 AI 専用フォーマット
# -------------------------------------------------------

def notify_daily_summary(summary: dict) -> bool:
    """
    日次損益サマリーを LINE に送信する。

    Parameters
    ----------
    summary : auto.recorder.daily_summary() の戻り値
    """
    d         = summary["date"]
    date_fmt  = f"{d[:4]}/{d[4:6]}/{d[6:8]}"

    profit    = summary["profit"]
    profit_sign = "📈" if profit >= 0 else "📉"

    roi       = summary["roi_pct"]
    hit       = summary["hit_count"]
    hit_rate  = summary["hit_rate_pct"]
    races_bet = summary["races_bet"]
    races_skip= summary["races_skip"]

    lines = [
        f"🚤 競艇 AI 日次レポート",
        f"📅 {date_fmt}",
        "",
        f"▼ レース",
        f"  ベットあり : {races_bet} レース",
        f"  見送り     : {races_skip} レース",
        f"  的中       : {hit} 回  ({hit_rate:.1f}%)",
        "",
        f"▼ 損益",
        f"  賭け金     : {summary['total_bet']:,} 円",
        f"  払戻       : {summary['total_return']:,} 円",
        f"  損益       : {profit:+,} 円  {profit_sign}",
        f"  回収率     : {roi:.1f} %",
        "",
        f"⏰ {datetime.now().strftime('%H:%M')} 集計",
    ]
    text = "\n".join(lines)

    print("\n--- LINE 日次レポート ---")
    print(text)
    print("------------------------")

    ok = send_line_message(text)
    print("[LINE] 送信成功" if ok else "[LINE] 送信失敗（.env を確認してください）")
    return ok


def notify_bet_signal(race: dict, bets: list[dict], total_bet: int) -> bool:
    """
    ベット候補を LINE に通知する（オプション）。

    Parameters
    ----------
    race      : {"venue", "rno", "stime"}
    bets      : [{"combo", "amount", "ev_pct", "odds"}, ...]
    total_bet : 合計賭け金
    """
    lines = [
        f"🚤 競艇 AI ベット候補",
        f"📍 {race['venue']} {race['rno']}R  {race.get('stime','')}",
        "",
    ]
    for b in bets:
        ev_str = f"EV {b['ev_pct']:+.1f}%" if b.get("ev_pct") is not None else ""
        odds_str = f"@{b['odds']:.1f}" if b.get("odds") else ""
        lines.append(f"  {b['combo']}  {b['amount']:,}円  {odds_str}  {ev_str}")

    lines += ["", f"合計: {total_bet:,}円  (ドライラン・実投票なし)"]
    return send_line_message("\n".join(lines))


# -------------------------------------------------------
# 動作確認用
# -------------------------------------------------------

if __name__ == "__main__":
    ok = send_line_message(
        f"🚤 競艇 AI 通知テスト\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "LINE 通知の設定が完了しました！"
    )
    import sys
    sys.exit(0 if ok else 1)

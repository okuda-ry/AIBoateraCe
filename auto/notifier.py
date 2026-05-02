"""
auto/notifier.py — LINE Messaging API / Discord Webhook による通知モジュール。

設定方法:
  プロジェクトルートに .env ファイルを作成し、以下を記載:

    LINE_CHANNEL_ACCESS_TOKEN=<チャネルアクセストークン（長期）>
    LINE_USER_ID=<自分のユーザーID (Uxxxxxxxx...)>

  ※ Autrader プロジェクトの .env をそのままコピーして使用できます。
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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


def _get_discord_webhook_url() -> str:
    """Discord Webhook URL。未設定なら空文字を返す。"""
    return os.environ.get("DISCORD_WEBHOOK_URL", "").strip()


def _discord_notify_jcds() -> set[str]:
    """
    Discord の通知対象場コード。

    デフォルトは平和島のみ（jcd=04）。全場通知したい場合は
    DISCORD_NOTIFY_JCDS=* を指定する。
    """
    raw = os.environ.get("DISCORD_NOTIFY_JCDS", "04").strip()
    if raw == "*":
        return {"*"}
    return {v.strip().zfill(2) for v in raw.split(",") if v.strip()}


def should_notify_discord_race(race: dict) -> bool:
    """設定された場コードに該当するレースだけ Discord 通知する。"""
    targets = _discord_notify_jcds()
    return "*" in targets or str(race.get("jcd", "")).zfill(2) in targets


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


def send_discord_message(
    content: str = "",
    embeds: list[dict] | None = None,
    username: str = "競艇AI",
) -> bool:
    """
    Discord Webhook にメッセージを送信する。

    Webhook URL 未設定時は処理を止めずに False を返す。
    """
    webhook_url = _get_discord_webhook_url()
    if not webhook_url:
        print("[Discord] DISCORD_WEBHOOK_URL 未設定のためスキップ")
        return False

    payload: dict = {"username": username}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    if "content" not in payload and "embeds" not in payload:
        raise ValueError("Discord 送信には content または embeds が必要です")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AiBoateraCe/1.0 (+https://discord.com/api/webhooks)",
        },
        method="POST",
    )

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return True
                print(f"[Discord] HTTP {resp.status}")
                return False
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                retry_after = e.headers.get("Retry-After", "1")
                try:
                    wait_sec = min(float(retry_after), 5.0)
                except ValueError:
                    wait_sec = 1.0
                print(f"[Discord] rate limited. retry after {wait_sec:.1f}s")
                time.sleep(wait_sec)
                continue
            print(f"[Discord] HTTPError {e.code}: {e.read().decode(errors='ignore')}")
            return False
        except Exception as e:
            print(f"[Discord] 送信エラー: {e}")
            return False
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


def _format_boat_probabilities(prediction_result: dict) -> str:
    boats = prediction_result.get("boats_by_lane") or prediction_result.get("boats") or []
    if not boats:
        return "取得できませんでした"

    lines = []
    for boat in sorted(boats, key=lambda b: b.get("lane", 99)):
        lane = boat.get("lane", "?")
        name = boat.get("name", "")
        prob = boat.get("prob_pct")
        rank = boat.get("pred_rank")
        prob_str = f"{prob:.1f}%" if isinstance(prob, (int, float)) else "--%"
        rank_str = f"予想{rank}位" if rank else "予想-"
        lines.append(f"{lane}号艇 {name}  {prob_str}  ({rank_str})")
    return "\n".join(lines)


def _format_trifecta_top10(prediction_result: dict) -> str:
    trifecta = prediction_result.get("trifecta") or []
    if not trifecta:
        return "取得できませんでした"

    lines = []
    for row in trifecta[:10]:
        rank = row.get("rank", len(lines) + 1)
        combo = row.get("combo", "---")
        prob = row.get("prob_pct")
        odds = row.get("odds")
        ev = row.get("ev_pct")
        prob_str = f"{prob:.2f}%" if isinstance(prob, (int, float)) else "--%"
        odds_str = f"@{odds:.1f}" if isinstance(odds, (int, float)) else "@--"
        ev_str = f"EV {ev:+.1f}%" if isinstance(ev, (int, float)) else "EV --"
        lines.append(f"{rank:>2}. {combo:<5}  {prob_str:>6}  {odds_str:>6}  {ev_str}")
    return "\n".join(lines)


def notify_discord_prediction(
    race: dict,
    strategy_bets: dict,
    odds_dict: dict,
    probs_120=None,
    confidence: float | None = None,
    budget: int | None = None,
    min_edge: float | None = None,
    prediction_result: dict | None = None,
) -> bool:
    """
    予測完了後、各艇の1着確率と3連単上位10を Discord に通知する。

    デフォルトでは平和島（jcd=04）のみ通知する。
    """
    if not should_notify_discord_race(race):
        return False

    prediction_result = prediction_result or {}
    stime = race.get("stime", "")
    venue = race.get("venue", "")
    rno = race.get("rno", "")
    hd = race.get("hd", "")
    date_fmt = f"{hd[:4]}/{hd[4:6]}/{hd[6:8]}" if len(hd) == 8 else hd

    summary = [f"📍 {venue} {rno}R  {stime}発走", f"📅 {date_fmt}"]
    if confidence is not None:
        summary.append(f"信頼度 {confidence * 100:.1f}%")
    if prediction_result.get("has_beforeinfo") is not None:
        beforeinfo = "あり" if prediction_result.get("has_beforeinfo") else "なし"
        summary.append(f"直前情報 {beforeinfo}")

    embed = {
        "title": "🚤 競艇AI 予測通知",
        "description": "\n".join(summary),
        "color": 0x58A6FF,
        "fields": [
            {
                "name": "各艇の1着確率",
                "value": _format_boat_probabilities(prediction_result),
                "inline": False,
            },
            {
                "name": "3連単 上位10",
                "value": f"```text\n{_format_trifecta_top10(prediction_result)}\n```",
                "inline": False,
            },
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return send_discord_message(embeds=[embed])


# -------------------------------------------------------
# 動作確認用
# -------------------------------------------------------

if __name__ == "__main__":
    test_text = (
        f"🚤 競艇 AI 通知テスト\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "通知設定が完了しました！"
    )
    line_ok = send_line_message(test_text)
    discord_ok = send_discord_message(content=test_text) if _get_discord_webhook_url() else False
    import sys
    sys.exit(0 if line_ok or discord_ok else 1)

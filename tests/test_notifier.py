"""
tests/test_notifier.py — Discord 通知の単体テスト。
"""

import json


def test_should_notify_discord_race_defaults_to_heiwajima(monkeypatch):
    import auto.notifier as notifier

    monkeypatch.delenv("DISCORD_NOTIFY_JCDS", raising=False)

    assert notifier.should_notify_discord_race({"jcd": "04"})
    assert not notifier.should_notify_discord_race({"jcd": "05"})


def test_should_notify_discord_race_accepts_all(monkeypatch):
    import auto.notifier as notifier

    monkeypatch.setenv("DISCORD_NOTIFY_JCDS", "*")

    assert notifier.should_notify_discord_race({"jcd": "01"})
    assert notifier.should_notify_discord_race({"jcd": "24"})


def test_notify_discord_prediction_posts_embed(monkeypatch, sample_race, biased_probs, biased_odds):
    import auto.notifier as notifier

    sent = {}

    class FakeResponse:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["timeout"] = timeout
        sent["headers"] = dict(req.header_items())
        sent["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test/token")
    monkeypatch.setenv("DISCORD_NOTIFY_JCDS", "04")
    monkeypatch.setattr(notifier.urllib.request, "urlopen", fake_urlopen)

    ok = notifier.notify_discord_prediction(
        race=sample_race,
        strategy_bets={
            "kelly": {"1-2-3": 300},
            "ip": {},
        },
        odds_dict=biased_odds,
        probs_120=biased_probs,
        confidence=0.42,
        budget=1000,
        min_edge=0.05,
        prediction_result={
            "boats_by_lane": [
                {"lane": 1, "name": "選手A", "prob_pct": 38.2, "pred_rank": 1},
                {"lane": 2, "name": "選手B", "prob_pct": 20.1, "pred_rank": 2},
            ],
            "trifecta": [
                {"rank": 1, "combo": "1-2-3", "prob_pct": 8.12, "odds": 15.2, "ev_pct": 23.4},
                {"rank": 2, "combo": "1-3-2", "prob_pct": 6.54, "odds": 18.5, "ev_pct": 21.0},
            ],
            "has_beforeinfo": True,
        },
    )

    assert ok
    assert sent["url"] == "https://discord.com/api/webhooks/test/token"
    assert req_header(sent, "User-agent").startswith("AiBoateraCe/")
    payload = sent["payload"]
    assert payload["username"] == "競艇AI"
    embed = payload["embeds"][0]
    assert embed["title"] == "🚤 競艇AI 予測通知"
    assert "平和島 1R" in embed["description"]
    assert "直前情報 あり" in embed["description"]
    assert embed["fields"][0]["name"] == "各艇の1着確率"
    assert "1号艇 選手A  38.2%" in embed["fields"][0]["value"]
    assert embed["fields"][1]["name"] == "3連単 上位10"
    assert "1-2-3" in embed["fields"][1]["value"]
    assert "EV +23.4%" in embed["fields"][1]["value"]


def req_header(sent, name):
    return sent.get("headers", {}).get(name) or sent.get("headers", {}).get(name.title(), "")


def test_notify_discord_prediction_skips_other_venues(monkeypatch, sample_race):
    import auto.notifier as notifier

    called = False

    def fake_send(*args, **kwargs):
        nonlocal called
        called = True
        return True

    race = {**sample_race, "jcd": "05", "venue": "多摩川"}
    monkeypatch.setenv("DISCORD_NOTIFY_JCDS", "04")
    monkeypatch.setattr(notifier, "send_discord_message", fake_send)

    ok = notifier.notify_discord_prediction(
        race=race,
        strategy_bets={"kelly": {"1-2-3": 100}},
        odds_dict={},
    )

    assert not ok
    assert not called

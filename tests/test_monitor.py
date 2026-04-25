"""
tests/test_monitor.py — monitor.py (Flask Blueprint) のルートテスト。

Flask テストクライアントを使い、実際の HTTP リクエストをシミュレートする。
DB_PATH をモンキーパッチして本番 DB に依存しない。
"""

import sqlite3
from pathlib import Path

import pytest


# -------------------------------------------------------
# Flask アプリのセットアップ
# -------------------------------------------------------

@pytest.fixture
def app(monkeypatch, tmp_path):
    """
    テスト用 Flask アプリを生成する。
    monitor.DB_PATH を一時 DB に差し替えてから Blueprint を登録する。
    """
    db_file = tmp_path / "monitor_test.db"

    # recorder と monitor が参照する DB_PATH をパッチ
    import auto.recorder as rec
    import monitor as mon

    monkeypatch.setattr(rec, "DB_PATH", db_file)
    monkeypatch.setattr(mon, "DB_PATH", db_file)
    # _db_initialized リセット（各テストで再初期化）
    monkeypatch.setattr(mon, "_db_initialized", False)

    rec.init_db()

    from flask import Flask, render_template_string
    flask_app = Flask(__name__, template_folder="../templates")
    flask_app.secret_key = "test"

    # base.html の url_for('index') / race_detail の url_for('predict') 用ダミールート
    @flask_app.route("/")
    def index():
        return "ok"

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        return "ok"

    flask_app.register_blueprint(mon.monitor_bp)

    flask_app.config["TESTING"] = True
    yield flask_app, db_file


@pytest.fixture
def client(app):
    flask_app, db_file = app
    with flask_app.test_client() as c:
        yield c, db_file


# -------------------------------------------------------
# /monitor  ダッシュボード
# -------------------------------------------------------

class TestDashboard:
    def test_empty_db_returns_200(self, client):
        c, _ = client
        resp = c.get("/monitor/")
        assert resp.status_code == 200

    def test_no_db_returns_200(self, monkeypatch, tmp_path):
        """DB ファイルが存在しない場合でも 200 を返す。"""
        import auto.recorder as rec
        import monitor as mon

        nonexistent = tmp_path / "nonexistent.db"
        monkeypatch.setattr(rec, "DB_PATH", nonexistent)
        monkeypatch.setattr(mon, "DB_PATH", nonexistent)
        monkeypatch.setattr(mon, "_db_initialized", False)

        from flask import Flask
        flask_app = Flask(__name__, template_folder="../templates")
        flask_app.secret_key = "test"

        @flask_app.route("/")
        def index():
            return "ok"

        @flask_app.route("/predict", methods=["POST"])
        def predict():
            return "ok"

        flask_app.register_blueprint(mon.monitor_bp)
        flask_app.config["TESTING"] = True

        with flask_app.test_client() as c:
            resp = c.get("/monitor/")
        assert resp.status_code == 200

    def test_html_contains_monitor(self, client):
        c, _ = client
        resp = c.get("/monitor/")
        assert b"monitor" in resp.data.lower() or resp.status_code == 200


# -------------------------------------------------------
# /monitor/races
# -------------------------------------------------------

class TestRacesList:
    def test_empty_returns_200(self, client):
        c, _ = client
        resp = c.get("/monitor/races?hd=20260406")
        assert resp.status_code == 200

    def test_with_data_returns_200(self, client):
        c, db_file = client
        import auto.recorder as rec
        rec.upsert_race({
            "race_id": "20260406_04_01",
            "jcd": "04", "venue": "平和島",
            "hd": "20260406", "rno": 1, "stime": "10:00",
        })
        resp = c.get("/monitor/races?hd=20260406")
        assert resp.status_code == 200

    def test_races_list_shows_all_strategy_columns(self, client, biased_odds):
        c, db_file = client
        import auto.recorder as rec
        race = {
            "race_id": "20260406_04_04",
            "jcd": "04", "venue": "平和島",
            "hd": "20260406", "rno": 4, "stime": "11:30",
        }
        rec.upsert_race(race)
        rec.save_strategy_bets(race["race_id"], "strict_flat", {"1-2-3": 100}, biased_odds)
        rec.update_result(race["race_id"], "1-2-3", 3530)

        resp = c.get("/monitor/races?hd=20260406")
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "strict_flat" in html
        assert "true_kelly_cap" in html
        assert "+3,430" in html

    def test_hd_param_used(self, client):
        """hd パラメータが違う日付でも 200 を返す。"""
        c, _ = client
        resp = c.get("/monitor/races?hd=20991231")
        assert resp.status_code == 200


# -------------------------------------------------------
# /monitor/race/<race_id>
# -------------------------------------------------------

class TestRaceDetail:
    def test_nonexistent_race_returns_200(self, client):
        """存在しない race_id でも 200 を返す（race=None の分岐）。"""
        c, _ = client
        resp = c.get("/monitor/race/NONEXISTENT")
        assert resp.status_code == 200

    def test_existing_race_returns_200(self, client):
        c, db_file = client
        import auto.recorder as rec
        race = {
            "race_id": "20260406_04_02",
            "jcd": "04", "venue": "平和島",
            "hd": "20260406", "rno": 2, "stime": "10:30",
        }
        rec.upsert_race(race)
        resp = c.get(f"/monitor/race/{race['race_id']}")
        assert resp.status_code == 200

    def test_race_with_bets_returns_200(self, client, biased_odds):
        c, db_file = client
        import auto.recorder as rec
        race = {
            "race_id": "20260406_04_03",
            "jcd": "04", "venue": "平和島",
            "hd": "20260406", "rno": 3, "stime": "11:00",
        }
        rec.upsert_race(race)
        rec.save_strategy_bets(race["race_id"], "kelly", {"1-2-3": 300}, biased_odds)
        rec.save_strategy_bets(race["race_id"], "ip",    {"1-2-3": 500}, biased_odds)
        rec.update_result(race["race_id"], "1-2-3", 3530)
        resp = c.get(f"/monitor/race/{race['race_id']}")
        assert resp.status_code == 200


# -------------------------------------------------------
# /monitor/history
# -------------------------------------------------------

class TestHistory:
    def test_empty_returns_200(self, client):
        c, _ = client
        resp = c.get("/monitor/history")
        assert resp.status_code == 200

    def test_days_param(self, client):
        c, _ = client
        resp = c.get("/monitor/history?days=30")
        assert resp.status_code == 200

    def test_with_data_returns_200(self, client, biased_odds):
        c, db_file = client
        import auto.recorder as rec
        race = {
            "race_id": "20260406_04_01",
            "jcd": "04", "venue": "平和島",
            "hd": "20260406", "rno": 1, "stime": "10:00",
        }
        rec.upsert_race(race)
        rec.save_strategy_bets(race["race_id"], "kelly", {"1-2-3": 200}, biased_odds)
        rec.update_result(race["race_id"], "1-2-3", 3530)
        resp = c.get("/monitor/history?days=7")
        assert resp.status_code == 200

"""
launch.py — 競艇AI 起動スクリプト。

デスクトップショートカットから呼ばれる。
  1. run_auto.py（スケジューラ）を新しいコンソールで起動
  2. app.py（Web UI）を新しいコンソールで起動
  3. Flask が実際に応答するまでポーリングしてからブラウザを開く

pythonw.exe で実行することでランチャー自体はコンソールを表示しない。
"""

import subprocess
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).parent
PYTHON  = Path("C:/venv/boatrace/Scripts/python.exe")
URL     = "http://localhost:5000/monitor"

# ---- run_auto.py ----
subprocess.Popen(
    [str(PYTHON), "run_auto.py"],
    cwd=str(PROJECT),
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)

# ---- app.py ----
subprocess.Popen(
    [str(PYTHON), "app.py"],
    cwd=str(PROJECT),
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)

# Flask が起動するまでポーリング（最大 60 秒）
for _ in range(60):
    time.sleep(1)
    try:
        urllib.request.urlopen(URL, timeout=1)
        break  # 応答があればループを抜ける
    except (urllib.error.URLError, OSError):
        continue  # まだ起動していない → 待つ

webbrowser.open(URL)

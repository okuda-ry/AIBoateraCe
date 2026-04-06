"""
launch.py — 競艇AI 起動スクリプト。

デスクトップショートカットから呼ばれる。
  1. run_auto.py（スケジューラ）を新しいコンソールで起動
  2. app.py（Web UI）を新しいコンソールで起動
  3. 3秒後にブラウザで http://localhost:5000/monitor を開く

pythonw.exe で実行することでランチャー自体はコンソールを表示しない。
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).parent
PYTHON  = Path("C:/venv/boatrace/Scripts/python.exe")

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

# Flask の起動を待ってからブラウザを開く
time.sleep(3)
webbrowser.open("http://localhost:5000/monitor")

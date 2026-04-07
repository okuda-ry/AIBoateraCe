@echo off
cd /d "%~dp0"

:: スケジューラ起動（新しいウィンドウ）
start "競艇AI スケジューラ" C:\venv\boatrace\Scripts\python.exe run_auto.py

:: Web UI 起動（新しいウィンドウ）
start "競艇AI Web UI" C:\venv\boatrace\Scripts\python.exe app.py

:: Flask が起動するまで待機してからブラウザを開く
C:\venv\boatrace\Scripts\python.exe -c "
import time, urllib.request, urllib.error, webbrowser
for _ in range(60):
    time.sleep(1)
    try:
        urllib.request.urlopen('http://localhost:5000/monitor', timeout=1)
        break
    except:
        pass
webbrowser.open('http://localhost:5000/monitor')
"

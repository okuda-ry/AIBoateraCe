"""
tools/setup_shortcut.py — デスクトップショートカット作成スクリプト。

一度だけ実行してください:
    C:\venv\boatrace\Scripts\python.exe tools/setup_shortcut.py

実行後、デスクトップに「競艇AI」ショートカットが作成されます。
"""

import subprocess
import sys
from pathlib import Path

PROJECT  = Path(__file__).parent.parent
TOOLS    = Path(__file__).parent
ICON_OUT = TOOLS / "boatrace.ico"
DESKTOP  = Path.home() / "OneDrive" / "デスクトップ"
PYTHONW  = Path("C:/venv/boatrace/Scripts/pythonw.exe")
LAUNCH   = PROJECT / "launch.py"
SHORTCUT = DESKTOP / "競艇AI.lnk"


# -------------------------------------------------------
# Step 1: ピクセルアートアイコンを生成
# -------------------------------------------------------

def make_icon():
    try:
        from PIL import Image
    except ImportError:
        print("[setup] Pillow をインストール中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image

    # ---- カラーパレット ----
    S = (135, 206, 235)   # 空（水色）
    D = (10,  70,  160)   # 深い水（濃紺）
    M = (25,  105, 195)   # 中間の水
    L = (70,  150, 225)   # 波ハイライト
    W = (248, 248, 252)   # 船体（白）
    K = (28,  36,  56)    # アウトライン（ダーク）
    G = (175, 185, 205)   # キャビン（グレー）
    V = (155, 210, 250)   # 窓（明るい水色）
    P = (220, 238, 255)   # 航跡スプレー
    R = (210, 45,  35)    # 赤ストライプ
    Y = (240, 190, 30)    # 黄色アクセント

    # 32x32 グリッド（デフォルト: 空）
    g = [[S] * 32 for _ in range(32)]

    # ---- 水面（row 18〜31）----
    for y in range(18, 32):
        for x in range(32):
            g[y][x] = D

    # ---- 波パターン ----
    # 規則的なハイライトで波を表現
    wave_rows = [20, 23, 26, 29]
    for wy in wave_rows:
        for x in range(0, 32, 5):
            for dx in range(3):
                if x + dx < 32:
                    g[wy][x + dx] = L

    # ---- 航跡・スプレー（船の後方・左側）----
    for y in range(16, 21):
        for x in range(0, 8):
            dist = x + (y - 16)
            if dist < 10:
                g[y][x] = P if (x + y) % 2 == 0 else L

    # ---- 船体ライン（row 15〜18, col 5〜27）----
    # 赤ストライプ（row 15）
    for x in range(6, 27):
        g[15][x] = R
    g[15][5]  = K
    g[15][27] = K

    # 白い船体（row 16〜17）
    for x in range(5, 28):
        g[16][x] = W
    for x in range(6, 27):
        g[17][x] = W

    # 船体アウトライン
    g[16][4]  = K;  g[16][28] = K
    g[17][5]  = K;  g[17][27] = K

    # 船底（row 18、アウトライン）
    for x in range(5, 28):
        g[18][x] = K

    # 船首（右側を尖らせる）
    g[14][26] = K
    g[15][27] = K;  g[15][28] = K
    g[16][28] = K;  g[16][29] = K
    g[17][27] = K;  g[17][28] = W

    # ---- キャビン（row 10〜15, col 9〜19）----
    for y in range(10, 15):
        for x in range(9, 20):
            g[y][x] = G

    # キャビンアウトライン
    for x in range(9, 20):
        g[10][x] = K
        g[14][x] = K
    for y in range(10, 15):
        g[y][9]  = K
        g[y][19] = K

    # キャビン上部（屋根のアクセント）
    for x in range(10, 19):
        g[10][x] = K
    for x in range(11, 18):
        g[9][x] = K

    # ---- 窓（2つ）----
    for y in range(11, 14):
        # 左窓
        for x in range(11, 14):
            g[y][x] = V
        # 右窓
        for x in range(15, 18):
            g[y][x] = V

    # 窓枠
    for x in [11, 13, 15, 17]:
        g[11][x] = K; g[13][x] = K
    for y in [11, 13]:
        for x in range(11, 14):
            g[y][x] = K
        for x in range(15, 18):
            g[y][x] = K

    # ---- 黄色ナンバー「1」（キャビン前面）----
    # 小さい "1" を col 21〜23、row 12〜16 に描く
    for y in range(12, 16):
        g[y][22] = Y
    g[12][21] = Y

    # ---- 画像生成 ----
    img32 = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(32):
        for x in range(32):
            img32.putpixel((x, y), g[y][x] + (255,))

    # ICO 保存（16, 32, 48px の3サイズ）
    TOOLS.mkdir(exist_ok=True)
    img32.save(str(ICON_OUT), format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"[setup] アイコン作成完了: {ICON_OUT}")
    return ICON_OUT


# -------------------------------------------------------
# Step 2: デスクトップにショートカットを作成
# -------------------------------------------------------

def make_shortcut(icon_path: Path):
    if not DESKTOP.exists():
        print(f"[setup] デスクトップが見つかりません: {DESKTOP}")
        return

    # パスをバックスラッシュに統一（PowerShell 用）
    lnk      = str(SHORTCUT).replace("/", "\\")
    target   = str(PYTHONW).replace("/", "\\")
    args     = f'"{str(LAUNCH).replace("/", "\\")}"'
    workdir  = str(PROJECT).replace("/", "\\")
    ico      = str(icon_path).replace("/", "\\")

    # PowerShell スクリプトを一時ファイルに書き出してから実行
    # （日本語パスを含む場合に -Command 直渡しが文字化けすることを回避）
    ps_lines = [
        "$WshShell = New-Object -ComObject WScript.Shell",
        f'$sc = $WshShell.CreateShortcut("{lnk}")',
        f'$sc.TargetPath       = "{target}"',
        f'$sc.Arguments        = \'{args}\'',
        f'$sc.WorkingDirectory = "{workdir}"',
        f'$sc.IconLocation     = "{ico}"',
        '$sc.Description      = "競艇AI — ドライランモニター起動"',
        "$sc.Save()",
        'Write-Host "OK"',
    ]
    ps_file = TOOLS / "_tmp_shortcut.ps1"
    ps_file.write_text("\r\n".join(ps_lines), encoding="utf-8-sig")

    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(ps_file)],
        capture_output=True, text=True, encoding="utf-8"
    )
    ps_file.unlink(missing_ok=True)

    if result.returncode == 0 and "OK" in result.stdout:
        print(f"[setup] ショートカット作成完了: {SHORTCUT}")
    else:
        print(f"[setup] ショートカット作成失敗:")
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr)


# -------------------------------------------------------
# メイン
# -------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  競艇AI デスクトップショートカット セットアップ")
    print("=" * 50)

    icon = make_icon()
    make_shortcut(icon)

    print()
    print("完了! デスクトップの「競艇AI」をダブルクリックで起動できます。")
    print("  → スケジューラと Web UI が起動し、ブラウザが自動で開きます。")

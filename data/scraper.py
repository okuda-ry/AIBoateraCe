"""
boatrace.jp からリアルタイムデータをスクレイピングし、
モデル用の特徴量を生成するモジュール。

対応 URL 形式:
    1. 平和島場サイト:
       https://www.heiwajima.gr.jp/asp/heiwajima/kyogi/kyogihtml/index.htm?racenum=1
    2. 公式出走表:
       https://www.boatrace.jp/owpc/pc/race/racelist?jcd=04&hd=20260320&rno=1
    3. 公式直前情報:
       https://www.boatrace.jp/owpc/pc/race/beforeinfo?jcd=04&hd=20260320&rno=1

使い方:
    from data.scraper import scrape_race
    boat_feats, race_feats = scrape_race("https://...url...", race_columns)
"""

import re
from datetime import date
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------
# 定数
# -------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 会場名 → 公式サイト jcd コード
VENUE_JCD = {
    "heiwajima": "04", "平和島": "04",
    "kiryu": "01",     "桐生":   "01",
    "toda": "02",      "戸田":   "02",
    "edogawa": "03",   "江戸川": "03",
    "tamagawa": "05",  "多摩川": "05",
    "hamanako": "06",  "浜名湖": "06",
    "gamagori": "07",  "蒲郡":   "07",
    "tokoname": "08",  "常滑":   "08",
    "tsu": "09",       "津":     "09",
    "mikuni": "10",    "三国":   "10",
    "biwako": "11",    "びわこ": "11",
    "suminoe": "12",   "住之江": "12",
    "amagasaki": "13", "尼崎":   "13",
    "naruto": "14",    "鳴門":   "14",
    "marugame": "15",  "丸亀":   "15",
    "kojima": "16",    "児島":   "16",
    "hiroshima": "17", "宮島":   "17",
    "tokuyama": "18",  "徳山":   "18",
    "shimonoseki":"19","下関":   "19",
    "wakamatsu": "20", "若松":   "20",
    "ashiya": "21",    "芦屋":   "21",
    "fukuoka": "22",   "福岡":   "22",
    "karatsu": "23",   "唐津":   "23",
    "omura": "24",     "大村":   "24",
}

KYUBETSU_MAP = {"A1": 3.0, "A2": 2.0, "B1": 1.0, "B2": 0.0}

# 今節成績スロット: 今節成績_{i}-{j}  i=1-6, j=1-2
TODAY_SLOTS = [f"{i}-{j}" for i in range(1, 7) for j in range(1, 3)]

# 艇ごとの数値特徴量リスト（preprocess.py と合わせること）
BOAT_NUM_FEATS = [
    "年齢", "体重", "全国勝率", "全国2連対率",
    "当地勝率", "当地2連対率", "モーター2連対率", "ボート2連対率", "早見",
    "今節平均ST",  # 今節の平均スタートタイム (小さいほど早い = 有利)
]
BOAT_TODAY_FEATS = [f"今節成績_{s}" for s in TODAY_SLOTS]


# -------------------------------------------------------
# URL パース
# -------------------------------------------------------

def _url_to_jcd_hd_rno(url: str) -> tuple[str, str, int]:
    """任意の対応 URL から (jcd, hd, rno) を返す。"""
    qs = parse_qs(urlparse(url).query)

    if "heiwajima.gr.jp" in url:
        jcd = "04"
        hd  = date.today().strftime("%Y%m%d")
        rno = int(qs.get("racenum", ["1"])[0])
    elif "boatrace.jp" in url:
        jcd = qs.get("jcd", ["04"])[0]
        hd  = qs.get("hd",  [date.today().strftime("%Y%m%d")])[0]
        rno = int(qs.get("rno", ["1"])[0])
    else:
        raise ValueError(f"未対応のURL形式: {url}\n"
                         "対応: heiwajima.gr.jp / boatrace.jp")

    return jcd, hd, rno


# -------------------------------------------------------
# HTTP フェッチ
# -------------------------------------------------------

def _fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return BeautifulSoup(resp.text, "html.parser")


# -------------------------------------------------------
# テキスト → 数値変換ユーティリティ
# -------------------------------------------------------

def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return default


def _result_code(s: str) -> float:
    """今節成績コードを数値に変換。F/L/K/転/妨 → 7.0 、未出走 → 0.0"""
    s = str(s).strip()
    if re.match(r"^[1-6]$", s):
        return float(s)
    if s in ("F", "Ｆ", "L", "Ｌ", "K", "Ｋ", "転", "妨", "S", "Ｓ"):
        return 7.0
    return 0.0


# -------------------------------------------------------
# 出走表ページ (racelist) パーサ
# -------------------------------------------------------

def _parse_racelist(soup: BeautifulSoup) -> list[dict]:
    """
    出走表ページから 6 艇分のデータを抽出する。

    boatrace.jp の出走表は <tbody> が艇ごとに分かれており、
    最初の <tr> に主要な数値が並んでいる。

    Returns
    -------
    boats : list of 6 dict — 枠番 1 始まり
    """
    boats = []
    for table in soup.find_all("table"):
        tbodies = table.find_all("tbody")
        if len(tbodies) < 6:
            continue

        candidate = []
        for tbody in tbodies[:6]:
            rows = tbody.find_all("tr")
            # 全セルのテキストをフラットに展開
            all_cells = [td.get_text(" ", strip=True)
                         for row in rows for td in row.find_all("td")]
            if not all_cells:
                continue
            candidate.append(all_cells)

        if len(candidate) == 6:
            # 妥当性チェック: 各艇の先頭セルが数字っぽいか
            if all(re.search(r"\d", c[0]) for c in candidate if c):
                boats = candidate
                break

    if not boats:
        raise RuntimeError(
            "出走表テーブルが見つかりませんでした。\n"
            "ページ構造が変わった可能性があります。--debug オプションで HTML を確認してください。"
        )

    result = []
    for lane, cells in enumerate(boats, start=1):
        d = _extract_player_cells(cells, lane)
        result.append(d)

    return result


def _extract_player_cells(cells: list[str], lane: int) -> dict:
    """
    1 艇分のセルリストから特徴量辞書を作る。

    boatrace.jp の列順（概略）:
        [0]  枠画像 or 登録番号
        [1]  選手名
        [2]  支部
        [3]  年齢
        [4]  体重
        [5]  級別
        [6]  F数
        [7]  L数
        [8]  平均ST  ← 早見として使用
        [9]  全国勝率
        [10] 全国2連対率
        [11] 当地勝率
        [12] 当地2連対率
        [13] モーター番号
        [14] モーター2連対率
        [15] ボート番号
        [16] ボート2連対率
        [17+] 今節成績 (複数セル)
    """
    def g(idx, default=0.0):
        return _to_float(cells[idx], default) if idx < len(cells) else default

    # 級別は文字列
    kyu_str = cells[5].strip() if len(cells) > 5 else ""
    kyu_val = KYUBETSU_MAP.get(kyu_str, 0.0)

    # 今節成績: セル17以降を順に読む（最大12スロット）
    today_results = {}
    for slot_idx, slot_key in enumerate(TODAY_SLOTS):
        cell_idx = 17 + slot_idx
        if cell_idx < len(cells):
            today_results[f"今節成績_{slot_key}"] = _result_code(cells[cell_idx])
        else:
            today_results[f"今節成績_{slot_key}"] = 0.0

    # 選手名: 漢字を含む最初の長めの文字列
    name = ""
    for c in cells[1:6]:
        if re.search(r"[ぁ-ん一-龯ァ-ヶ]", c) and len(c) >= 2:
            name = c
            break

    d = {
        "枠番":         lane,
        "選手名":       name,
        "年齢":         g(3),
        "体重":         g(4),
        "級別_val":     kyu_val,
        "全国勝率":     g(9),
        "全国2連対率":  g(10),
        "当地勝率":     g(11),
        "当地2連対率":  g(12),
        "モーター2連対率": g(14),
        "ボート2連対率":   g(16),
        "早見":         g(8),   # 平均ST を代替として使用
    }
    d.update(today_results)
    return d


# -------------------------------------------------------
# 直前情報ページ (beforeinfo) パーサ
# -------------------------------------------------------

def _parse_beforeinfo(soup: BeautifulSoup) -> tuple[dict, dict, dict]:
    """
    直前情報ページから 展示タイム・スタートタイム・気象情報 を抽出する。

    Returns
    -------
    tenji_times : {枠番(int): 展示タイム(float)}
    weather     : {天候, 風向, 風速, 波高, 距離} dict
    st_times    : {枠番(int): 平均ST(float)}  今節のST平均（取得できない場合は空）
    """
    tenji_times = {}
    weather = {}
    st_times = {}

    # --------- 展示タイム ---------
    # テキストパターン "6.88" が1行に並ぶテーブルを探す
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            # 艇番らしい数字 + タイム(6.xx) が混在する行を検出
            texts = [td.get_text(strip=True) for td in tds]
            # 6.xx パターンを探す
            times = [t for t in texts if re.match(r"^6\.\d{2}$", t)]
            if len(times) >= 1:
                # 枠番とタイムを対応付け
                lane = 0
                for t in texts:
                    if re.match(r"^[1-6]$", t):
                        lane = int(t)
                    elif re.match(r"^6\.\d{2}$", t) and lane > 0:
                        tenji_times[lane] = float(t)
                        lane = 0
                break

        if len(tenji_times) >= 6:
            break

    # 見つからない場合は全テキストからスキャン
    if not tenji_times:
        all_text = soup.get_text()
        for m in re.finditer(r"([1-6])\s*号?艇[^\n]*?(6\.\d{2})", all_text):
            lane_s, t_s = m.group(1), m.group(2)
            tenji_times[int(lane_s)] = float(t_s)

    # --------- スタートタイム（ST）---------
    # 直前情報ページには今節のST一覧が掲載されている場合がある
    # 形式: "0.00" ～ "0.39"（正常範囲）、"F" = フライング
    # ページ全体からST値を艇番と対応付けて収集する
    text = soup.get_text()

    # STパターン: "0.0x" ～ "0.3x" の小数（着順の小数とは範囲が異なる）
    st_pattern = re.compile(r'\b(0\.\d{2})\b')
    # 艇番パターンとST値を行ごとに走査して対応付ける
    st_lane_buf: dict[int, list[float]] = {}
    current_lane = 0
    for line in text.splitlines():
        line = line.strip()
        if re.match(r'^[1-6]$', line):
            current_lane = int(line)
        elif current_lane > 0:
            for m_st in st_pattern.finditer(line):
                st_val = float(m_st.group(1))
                if 0.0 <= st_val <= 0.50:   # STの現実的な範囲
                    st_lane_buf.setdefault(current_lane, []).append(st_val)
            # 次の艇番が来るまでリセットしない
            if re.match(r'^[1-6]$', line):
                current_lane = int(line)

    # 艇ごとのST平均を計算
    for lane, vals in st_lane_buf.items():
        if vals:
            st_times[lane] = round(float(np.mean(vals)), 3)

    # --------- 気象情報 ---------
    # 天候
    for pat, label in [("晴", "晴"), ("曇", "曇"), ("雨", "雨"), ("雪", "雪")]:
        if pat in text:
            weather.setdefault("天候", label)
            break

    # 風向
    for pat in ["北北東", "北東", "東北東", "東", "東南東", "南東",
                "南南東", "南", "南南西", "南西", "西南西", "西",
                "西北西", "北西", "北北西", "北"]:
        if pat in text:
            weather.setdefault("風向", pat)
            break

    # 風速 (数値 + m)
    m = re.search(r"風速[^\d]*(\d+\.?\d*)\s*m", text)
    if not m:
        m = re.search(r"(\d+\.?\d*)\s*m(?!\w)", text)
    weather["風速"] = float(m.group(1)) if m else 0.0

    # 波高 (数値 + cm)
    m = re.search(r"波[^\d]*(\d+\.?\d*)\s*cm", text)
    weather["波高"] = float(m.group(1)) if m else 0.0

    # 距離 (1800m が標準)
    m = re.search(r"(\d{4})\s*m", text)
    weather["距離"] = float(m.group(1)) if m else 1800.0

    return tenji_times, weather, st_times


# -------------------------------------------------------
# メイン: 特徴量生成
# -------------------------------------------------------

def scrape_race(url: str,
                race_columns: list[str],
                jcd: str = None,
                hd: str  = None,
                rno: int = None,
                venue_name: str = "平和島",
                nichiji: int = 1,
                debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    指定 URL のレースを scrape して (boat_features, race_features) を返す。

    Parameters
    ----------
    url          : 出走表 URL（heiwajima.gr.jp または boatrace.jp）
    race_columns : 学習時の race_features 列名リスト（保存ファイルから読む）
    jcd          : 会場コード（None の場合は url から自動取得）
    hd           : 日付 YYYYMMDD（None の場合は url or 今日）
    rno          : レース番号（None の場合は url から自動取得）
    venue_name   : レース場名（race_features の one-hot 用）
    nichiji      : 日次（何日目か）
    debug        : True にすると生テキストを表示

    Returns
    -------
    boat_features : (1, 6, 23)  float32
    race_features : (1, n_race_feats)  float32
    """
    if jcd is None or hd is None or rno is None:
        jcd, hd, rno = _url_to_jcd_hd_rno(url)

    racelist_url   = (f"https://www.boatrace.jp/owpc/pc/race/racelist"
                      f"?jcd={jcd}&hd={hd}&rno={rno:02d}")
    beforeinfo_url = (f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
                      f"?jcd={jcd}&hd={hd}&rno={rno:02d}")

    print(f"[scraper] 出走表取得中: {racelist_url}")
    soup_list = _fetch(racelist_url)

    if debug:
        print("[DEBUG] racelist text snippet:")
        print(soup_list.get_text()[:500])

    boats = _parse_racelist(soup_list)

    # 直前情報は任意（スタート展示前は取得できないため失敗しても続行）
    tenji_times, weather, st_times = {}, {}, {}
    try:
        print(f"[scraper] 直前情報取得中: {beforeinfo_url}")
        soup_before  = _fetch(beforeinfo_url)
        tenji_times, weather, st_times = _parse_beforeinfo(soup_before)
        if not tenji_times:
            print("[scraper] 展示タイム未公開（スタート展示前）— 出走表データのみで予測します")
        else:
            print(f"[scraper] 展示タイム取得: {tenji_times}")
        if st_times:
            print(f"[scraper] 今節ST取得: {st_times}")
    except Exception as e:
        print(f"[scraper] 直前情報を取得できませんでした（{e}）— 出走表データのみで予測します")

    # 展示タイムと今節STを各艇に統合
    for b in boats:
        lane = b["枠番"]
        if lane in tenji_times:
            b["早見"] = tenji_times[lane]
        # 今節平均ST（取得できなかった場合は 0.18 = 全国平均 を代入）
        b["今節平均ST"] = st_times.get(lane, 0.18)

    # -------- 艇特徴量 (1, 6, 23) --------
    boat_feat_matrix = []
    for b in sorted(boats, key=lambda x: x["枠番"]):
        row = (
            [b[f] for f in BOAT_NUM_FEATS]
            + [b[f"今節成績_{s}"] for s in TODAY_SLOTS]
            + [b["級別_val"]]
            + [float(b["枠番"])]
        )
        boat_feat_matrix.append(row)

    boat_features = np.array(boat_feat_matrix, dtype=np.float32)[np.newaxis, :]  # (1, 6, 23)

    # -------- レース特徴量 (1, n_race) --------
    # race_columns は学習時に保存したカラム名リスト
    # 数値部分
    num_vals = {
        "風速": weather.get("風速", 0.0),
        "波高": weather.get("波高", 0.0),
        "日次": float(nichiji),
        "距離": weather.get("距離", 1800.0),
    }

    # one-hot 部分 (天候, 風向, レース場)
    ohe_vals = {}
    for col in race_columns:
        if col.startswith("天候_"):
            ohe_vals[col] = 1.0 if col == f"天候_{weather.get('天候', '')}" else 0.0
        elif col.startswith("風向_"):
            ohe_vals[col] = 1.0 if col == f"風向_{weather.get('風向', '')}" else 0.0
        elif col.startswith("場_"):
            ohe_vals[col] = 1.0 if col == f"場_{venue_name}" else 0.0

    race_row = []
    for col in race_columns:
        if col in num_vals:
            race_row.append(num_vals[col])
        elif col in ohe_vals:
            race_row.append(ohe_vals[col])
        else:
            race_row.append(0.0)

    race_features = np.array([race_row], dtype=np.float32)  # (1, n_race)

    player_names    = [b.get("選手名", f"{b['枠番']}号艇")
                       for b in sorted(boats, key=lambda x: x["枠番"])]
    has_beforeinfo  = bool(tenji_times)   # 展示タイムが1件でも取得できたか

    print(f"[scraper] 艇特徴量: {boat_features.shape}  "
          f"レース特徴量: {race_features.shape}  "
          f"直前情報: {'あり' if has_beforeinfo else 'なし'}")

    return boat_features, race_features, player_names, weather, has_beforeinfo


# -------------------------------------------------------
# 3連単オッズ取得
# -------------------------------------------------------

# ページ上に表示される3連単オッズの順序
#
# boatrace.jp の odds3t テーブルは 6セクション（1着=1..6）が横並びの
# 1枚のHTMLテーブル。DOMを左→右→次行と読むと「行優先（row-major）」になる:
#
#   行p / 列first → combo
#   行0: [1-2-3, 2-1-3, 3-1-2, 4-1-2, 5-1-2, 6-1-2]
#   行1: [1-2-4, 2-1-4, 3-1-4, 4-1-3, 5-1-3, 6-1-3]
#   ...
#   行6: [1-3-5, ...]   ← flat index 36
#
# 各1着セクション内の順序:
#   2着は昇順(1着を除く), 3着は昇順(1着・2着を除く)
#   → セクション内位置 p: second = seconds[p//4], third = thirds[p%4]
#
def _build_odds3t_combo_order() -> list[str]:
    order = []
    for p in range(20):          # セクション内の位置 0..19
        for first in range(1, 7):    # 1着 (=HTMLの列)
            seconds = [x for x in range(1, 7) if x != first]
            second  = seconds[p // 4]
            thirds  = [x for x in range(1, 7) if x != first and x != second]
            third   = thirds[p % 4]
            order.append(f"{first}-{second}-{third}")
    return order

_ODDS3T_COMBO_ORDER: list[str] = _build_odds3t_combo_order()
# 検証: 120通りの全組み合わせを網羅しているはず
assert len(_ODDS3T_COMBO_ORDER) == 120
assert len(set(_ODDS3T_COMBO_ORDER)) == 120, "重複あり — マッピング定義を再確認してください"


def _parse_odds_text(text: str) -> float | None:
    """
    テキストをオッズ値としてパースする。

    対応表記:
      - 通常: "14.5" (小数点1桁)
      - 低配当: "1.5" ～ "6.4" (鉄板組み合わせ)
      - 高配当: "1084" (整数表記)
      - 更新中/未確定: "---" → None
    除外:
      - 艇番そのもの (1〜6 の整数1桁)
      - ページ番号・年号等 (5桁以上)

    Returns
    -------
    float if valid odds, None otherwise
    """
    text = text.strip()
    # 小数点1桁 (例: 14.5 / 1.5)
    if re.match(r'^\d{1,4}\.\d$', text):
        val = float(text)
        # 1.0 未満はオッズとして存在しない
        return val if val >= 1.0 else None
    # 整数2〜4桁 (高配当の整数表記 例: 1084)
    if re.match(r'^\d{2,4}$', text):
        val = float(text)
        return val if val >= 10 else None  # 10未満の整数は艇番等と区別できないので除外
    return None


def scrape_odds(jcd: str, hd: str, rno: int, debug: bool = False) -> dict:
    """
    boatrace.jp の3連単オッズページを取得する。

    テーブル構造:
      列  = 1着 (1〜6)
      行グループ = 2着 (5グループ × 4行 = 20行)
      セル値 = 3着オッズ

    DOM読み取り順（行優先）:
      行0: [1-2-3, 2-1-3, 3-1-2, 4-1-2, 5-1-2, 6-1-2]
      行1: [1-2-4, 2-1-4, 3-1-4, 4-1-3, 5-1-3, 6-1-3]
      ...

    Returns
    -------
    odds_dict : {combo_str: float}  例: {"1-2-3": 18.5}
                取得できない場合は空 dict（受付前/構造変更）。
    """
    url = (f"https://www.boatrace.jp/owpc/pc/race/odds3t"
           f"?jcd={jcd}&hd={hd}&rno={rno:02d}")

    try:
        soup = _fetch(url)
    except Exception as e:
        print(f"[scraper] オッズ取得失敗: {e}")
        return {}

    if debug:
        print(f"[DEBUG] odds3t URL: {url}")
        print(soup.get_text()[:3000])

    # ─── 方法1: テーブルの行構造を利用 ───────────────────────────
    # オッズテーブルの各 tr は 6列（1着ごとに1セル）
    # 1行あたりちょうど6値が揃う行のみをオッズ行として採用する。
    # 揃わない行（欠損・更新中セルあり）は None プレースホルダーで補完する。
    odds_values: list[float | None] = []
    found_by_method1 = False

    for table in soup.find_all("table"):
        candidate: list[float | None] = []
        valid_rows = 0
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            row_vals = [_parse_odds_text(td.get_text(strip=True)) for td in tds]
            # None でないものが 4つ以上あればオッズ行と判断
            non_null = [v for v in row_vals if v is not None]
            if len(non_null) >= 4:
                # 6列になるよう末尾を None で埋める
                row_6 = (row_vals + [None] * 6)[:6]
                candidate.extend(row_6)
                valid_rows += 1

        if valid_rows >= 18:   # 20行のうち18行以上取れていれば採用
            odds_values = candidate
            found_by_method1 = True
            if debug:
                non_null_count = sum(1 for v in odds_values if v is not None)
                print(f"[DEBUG] 方法1: {valid_rows}行取得  値={non_null_count}個")
            break

    # ─── 方法2: セル単位フラットスキャン（方法1で不足時）──────────
    if not found_by_method1:
        flat: list[float] = []
        for tag in soup.find_all(["td", "span"]):
            v = _parse_odds_text(tag.get_text(strip=True))
            if v is not None:
                flat.append(v)
        odds_values = flat  # type: ignore
        if debug:
            print(f"[DEBUG] 方法2: {len(odds_values)}個取得")

    # ─── combo_order と対応付け ──────────────────────────────────
    # None（欠損）はスキップして有効な値のみをマッピング
    odds_dict: dict[str, float] = {}
    valid_count = 0
    for combo, val in zip(_ODDS3T_COMBO_ORDER, odds_values):
        if val is not None:
            odds_dict[combo] = val
            valid_count += 1

    if odds_dict:
        vals = list(odds_dict.values())
        samples = ["1-2-3", "1-3-5", "3-1-2", "6-5-4"]
        sample_str = "  ".join(
            f"{c}={odds_dict[c]:.1f}" for c in samples if c in odds_dict
        )
        missing = 120 - valid_count
        missing_str = f"  欠損{missing}通り" if missing > 0 else ""
        print(f"[scraper] 3連単オッズ取得: {valid_count}通り{missing_str}  "
              f"min={min(vals):.1f}  max={max(vals):.1f}  "
              f"(検証: {sample_str})")
    else:
        print("[scraper] 3連単オッズ未取得（締切前/受付前/構造変更の可能性）")

    return odds_dict


# -------------------------------------------------------
# レーススケジュール取得
# -------------------------------------------------------

def scrape_schedule(hd: str = None) -> list[dict]:
    """
    指定日の全レーススケジュールを取得する。

    Parameters
    ----------
    hd : str  YYYYMMDD 形式。None の場合は今日。

    Returns
    -------
    races : [
        {"jcd": "04", "venue": "平和島", "hd": "20260405",
         "rno": 1, "stime": "10:00", "race_id": "04-20260405-01"},
        ...
    ]
    """
    if hd is None:
        hd = date.today().strftime("%Y%m%d")

    index_url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={hd}"
    try:
        soup = _fetch(index_url)
    except Exception as e:
        print(f"[scraper] スケジュール取得失敗: {e}")
        return []

    # 開催中の jcd を href から収集
    open_jcds: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r'[?&]jcd=(\d{2})', str(a["href"]))
        if m:
            open_jcds.add(m.group(1))

    # jcd → 日本語会場名マッピング（英字キーを除外）
    jcd_to_venue = {
        jcd: name
        for name, jcd in VENUE_JCD.items()
        if re.search(r'[\u3000-\u9fff]', name)   # 日本語文字を含む名前のみ
    }

    races: list[dict] = []
    for jcd in sorted(open_jcds):
        venue = jcd_to_venue.get(jcd, jcd)
        try:
            prog_url = (f"https://www.boatrace.jp/owpc/pc/race/raceindex"
                        f"?jcd={jcd}&hd={hd}")
            psoup = _fetch(prog_url)
            race_times = _parse_race_times(psoup)
            for rno, stime in race_times.items():
                races.append({
                    "jcd":     jcd,
                    "venue":   venue,
                    "hd":      hd,
                    "rno":     rno,
                    "stime":   stime,
                    "race_id": f"{jcd}-{hd}-{rno:02d}",
                })
        except Exception as e:
            print(f"[scraper] {venue}({jcd}) スケジュール取得失敗: {e}")

    races.sort(key=lambda r: (r["stime"], r["jcd"], r["rno"]))
    print(f"[scraper] スケジュール取得完了: {len(races)} レース / {len(open_jcds)} 場")
    return races


def _parse_race_times(soup: BeautifulSoup) -> dict[int, str]:
    """
    番組ページからレース番号 → 発走予定時刻の辞書を返す。

    Returns
    -------
    {1: "10:00", 2: "10:30", ...}
    """
    times: dict[int, str] = {}
    text = soup.get_text()

    # パターン1: "1R 10:00" / "1レース 10:00"
    for m in re.finditer(r'(\d{1,2})\s*[Rｒ]\s*(\d{1,2}:\d{2})', text):
        rno = int(m.group(1))
        t   = m.group(2)
        if 1 <= rno <= 12:
            times.setdefault(rno, t)

    # パターン2: テーブルから HH:MM を順番に抽出（上記で失敗した場合）
    if not times:
        time_cells = []
        for tag in soup.find_all(["td", "th", "span", "p", "div"]):
            raw = tag.get_text(strip=True)
            if re.match(r'^\d{1,2}:\d{2}$', raw):
                # ボートレースの発走時刻は 08:00〜21:00 程度
                h = int(raw.split(":")[0])
                if 8 <= h <= 21:
                    time_cells.append(raw)
        # 重複を除いて順番に割り当て
        seen: set[str] = set()
        rno = 1
        for t in time_cells:
            if t not in seen:
                times[rno] = t
                seen.add(t)
                rno += 1
            if rno > 12:
                break

    return times


# -------------------------------------------------------
# レース結果取得
# -------------------------------------------------------

def scrape_result(jcd: str, hd: str, rno: int) -> dict:
    """
    レース結果（3連単の組番と払戻金）を取得する。

    Returns
    -------
    {
        "combo":    "1-3-5",   # 3連単組番（未確定時は ""）
        "payout":   6740,      # 払戻金 (100円賭けた場合の払戻額)
        "finished": True,      # 結果確定フラグ
    }
    """
    url = (f"https://www.boatrace.jp/owpc/pc/race/raceresult"
           f"?jcd={jcd}&hd={hd}&rno={rno:02d}")
    try:
        soup = _fetch(url)
    except Exception as e:
        print(f"[scraper] 結果取得失敗: {e}")
        return {"combo": "", "payout": 0, "finished": False}

    text = soup.get_text()

    # ── 3連単組番を探す ──────────────────────────────────────
    # "1-3-5" 形式・全艇番が異なるものを採用
    combo = ""
    for m in re.finditer(r'\b([1-6])-([1-6])-([1-6])\b', text):
        a, b, c = m.group(1), m.group(2), m.group(3)
        if len({a, b, c}) == 3:
            combo = f"{a}-{b}-{c}"
            break

    # ── 払戻金を探す ─────────────────────────────────────────
    payout = 0

    # 方法1: "3連単" の直後に来る数字
    m = re.search(r'3連単[^\d]{0,20}?(\d{3,7})', text)
    if m:
        payout = int(m.group(1))

    # 方法2: 組番の直後に来る数字
    if payout == 0 and combo:
        pos = text.find(combo)
        if pos >= 0:
            after = text[pos: pos + 80]
            for m2 in re.finditer(r'(\d{3,7})', after):
                val = int(m2.group(1))
                if 100 <= val <= 9_999_900:
                    payout = val
                    break

    finished = bool(combo)
    if finished:
        print(f"[scraper] 結果: {combo}  払戻: {payout:,}円")
    else:
        print(f"[scraper] 結果未確定（レース未完了 or ページ構造変更）")

    return {"combo": combo, "payout": payout, "finished": finished}

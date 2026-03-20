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

def _parse_beforeinfo(soup: BeautifulSoup) -> tuple[dict, dict]:
    """
    直前情報ページから 展示タイム と 気象情報 を抽出する。

    Returns
    -------
    tenji_times : {枠番(int): 展示タイム(float)}
    weather     : {天候, 風向, 風速, 波高, 距離} dict
    """
    tenji_times = {}
    weather = {}

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

    # --------- 気象情報 ---------
    text = soup.get_text()

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

    return tenji_times, weather


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
    tenji_times, weather = {}, {}
    try:
        print(f"[scraper] 直前情報取得中: {beforeinfo_url}")
        soup_before  = _fetch(beforeinfo_url)
        tenji_times, weather = _parse_beforeinfo(soup_before)
        if not tenji_times:
            print("[scraper] 展示タイム未公開（スタート展示前）— 出走表データのみで予測します")
        else:
            print(f"[scraper] 展示タイム取得: {tenji_times}")
    except Exception as e:
        print(f"[scraper] 直前情報を取得できませんでした（{e}）— 出走表データのみで予測します")

    # 展示タイムを各艇に統合（beforeinfo の値で上書き）
    for b in boats:
        lane = b["枠番"]
        if lane in tenji_times:
            b["早見"] = tenji_times[lane]

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

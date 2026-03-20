"""
競艇公式サイトから LZH ファイルをダウンロードする。

使い方:
    python data/download.py racelists --start 2022-07-05 --end 2024-09-01
    python data/download.py results   --start 2022-05-13 --end 2024-09-01
"""

from datetime import datetime as dt, timedelta as td
from requests import get
from os import makedirs
from time import sleep

_URLS = {
    "racelists": "http://www1.mbrace.or.jp/od2/B/",
    "results":   "http://www1.mbrace.or.jp/od2/K/",
}
_DIRS = {
    "racelists": "downloads/racelists/lzh/",
    "results":   "downloads/results/lzh/",
}
_PREFIX = {
    "racelists": "b",
    "results":   "k",
}


def download(obj: str, start_date: str, end_date: str, interval: float = 1.0):
    """
    obj       : 'racelists' または 'results'
    start_date: 開始日 'YYYY-MM-DD'
    end_date  : 終了日 'YYYY-MM-DD'
    interval  : リクエスト間隔（秒）※サーバ負荷対策に 1 秒以上を推奨
    """
    assert obj in _URLS, f"obj は {list(_URLS.keys())} のいずれかを指定してください"

    save_dir = _DIRS[obj]
    makedirs(save_dir, exist_ok=True)

    prefix   = _PREFIX[obj]
    base_url = _URLS[obj]
    start    = dt.strptime(start_date, "%Y-%m-%d")
    end      = dt.strptime(end_date,   "%Y-%m-%d")

    print(f"ダウンロード開始: {start_date} 〜 {end_date}  ({(end - start).days + 1} 日分)")

    for d in range((end - start).days + 1):
        date_str = (start + td(days=d)).strftime("%Y%m%d")
        yyyymm   = date_str[:6]
        yymmdd   = date_str[2:]
        url      = f"{base_url}{yyyymm}/{prefix}{yymmdd}.lzh"
        dst      = f"{save_dir}{prefix}{yymmdd}.lzh"

        r = get(url)
        if r.status_code == 200:
            with open(dst, "wb") as f:
                f.write(r.content)
            print(f"  [OK] {url}")
        else:
            print(f"  [--] {url}  (HTTP {r.status_code})")

        sleep(interval)

    print("ダウンロード完了")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="競艇データ LZH ダウンロード")
    parser.add_argument("obj",      choices=["racelists", "results"], help="ダウンロード種別")
    parser.add_argument("--start",  default="2022-07-05",             help="開始日 YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-09-01",             help="終了日 YYYY-MM-DD")
    parser.add_argument("--interval", type=float, default=1.0,        help="リクエスト間隔（秒）")
    args = parser.parse_args()

    download(args.obj, args.start, args.end, args.interval)

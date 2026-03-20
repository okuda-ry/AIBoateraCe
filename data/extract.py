"""
LZH ファイルをテキストファイルに解凍する。

使い方:
    python data/extract.py racelists
    python data/extract.py results
"""

import os
import re
import lhafile

_LZH_DIRS = {
    "racelists": "downloads/racelists/lzh/",
    "results":   "downloads/results/lzh/",
}
_TXT_DIRS = {
    "racelists": "downloads/racelists/txt/",
    "results":   "downloads/results/txt/",
}


def extract(obj: str):
    """
    obj: 'racelists' または 'results'
    """
    assert obj in _LZH_DIRS, f"obj は {list(_LZH_DIRS.keys())} のいずれかを指定してください"

    lzh_dir = _LZH_DIRS[obj]
    txt_dir = _TXT_DIRS[obj]
    os.makedirs(txt_dir, exist_ok=True)

    lzh_files = sorted(f for f in os.listdir(lzh_dir) if re.search(r"\.lzh$", f, re.IGNORECASE))
    print(f"{len(lzh_files)} 個の LZH ファイルを解凍します → {txt_dir}")

    for filename in lzh_files:
        archive = lhafile.Lhafile(lzh_dir + filename)
        name    = archive.infolist()[0].filename
        open(txt_dir + name, "wb").write(archive.read(name))
        print(f"  [OK] {filename} → {name}")

    print("解凍完了")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LZH → TXT 解凍")
    parser.add_argument("obj", choices=["racelists", "results"], help="解凍対象")
    args = parser.parse_args()

    extract(args.obj)

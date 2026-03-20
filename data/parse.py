"""
テキストファイルを解析して CSV に変換する。

使い方:
    python data/parse.py timetable  --out downloads/racelists/csv/timetable.csv
    python data/parse.py details    --out downloads/results/details/details.csv
    python data/parse.py results    --out downloads/results/csv/results.csv
"""

import os
import re

# ========================================================
# 出走表（番組表）TXT → CSV
# ========================================================

_TIMETABLE_HEADER = (
    "タイトル,日次,レース日,レース場,"
    "レース回,レース名,距離(m),電話投票締切予定,"
    + ",".join(
        f"{n}枠_{col}"
        for n in range(1, 7)
        for col in [
            "艇番", "登録番号", "選手名", "年齢", "支部", "体重", "級別",
            "全国勝率", "全国2連対率", "当地勝率", "当地2連対率",
            "モーター番号", "モーター2連対率", "ボート番号", "ボート2連対率",
            "今節成績_1-1", "今節成績_1-2", "今節成績_2-1", "今節成績_2-2",
            "今節成績_3-1", "今節成績_3-2", "今節成績_4-1", "今節成績_4-2",
            "今節成績_5-1", "今節成績_5-2", "今節成績_6-1", "今節成績_6-2",
            "早見",
        ]
    )
    + "\n"
)


def _parse_timetable_file(text_file, csv_file):
    for contents in text_file:
        if re.search(r"番組表", contents):
            text_file.readline()
            title   = text_file.readline()[:-1].strip()
            text_file.readline()
            line    = text_file.readline()
            day     = line[3:7]
            date    = line[17:28]
            stadium = line[52:55]

        if re.search(r"電話投票締切予定", contents):
            line = contents
            if re.search(r"進入固定", line):
                line = line.replace("進入固定 Ｈ", "進入固定     Ｈ")
            race_round = line[0:3]
            race_name  = line[5:21]
            distance   = line[22:26]
            post_time  = line[37:42]

            for _ in range(4):
                text_file.readline()

            racer_data = ""
            line = text_file.readline()
            while line != "\n":
                if re.search(r"END", line):
                    break
                racer_data += (
                    "," + line[0]
                    + "," + line[2:6]
                    + "," + line[6:10]
                    + "," + line[10:12]
                    + "," + line[12:14]
                    + "," + line[14:16]
                    + "," + line[16:18]
                    + "," + line[19:23]
                    + "," + line[24:29]
                    + "," + line[30:34]
                    + "," + line[35:40]
                    + "," + line[41:43]
                    + "," + line[44:49]
                    + "," + line[50:52]
                    + "," + line[53:58]
                    + "," + line[59:60]
                    + "," + line[60:61]
                    + "," + line[61:62]
                    + "," + line[62:63]
                    + "," + line[63:64]
                    + "," + line[64:65]
                    + "," + line[65:66]
                    + "," + line[66:67]
                    + "," + line[67:68]
                    + "," + line[68:69]
                    + "," + line[69:70]
                    + "," + line[70:71]
                    + "," + line[71:73]
                )
                line = text_file.readline()

            csv_file.write(
                f"{title},{day},{date},{stadium},"
                f"{race_round},{race_name},{distance},{post_time}"
                + racer_data + "\n"
            )


def parse_timetable(txt_dir: str = "downloads/racelists/txt/",
                    csv_path: str = "downloads/racelists/csv/timetable.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="shift_jis") as f:
        f.write(_TIMETABLE_HEADER)

    txt_files = sorted(f for f in os.listdir(txt_dir) if re.search(r"\.TXT$", f, re.IGNORECASE))
    print(f"{len(txt_files)} 個のテキストファイルを処理します")

    for name in txt_files:
        with open(txt_dir + name, "r", encoding="shift_jis") as tf, \
             open(csv_path, "a", encoding="shift_jis") as cf:
            _parse_timetable_file(tf, cf)

    print(f"  → {csv_path}")


# ========================================================
# 競走成績（詳細）TXT → CSV
# ========================================================

_DETAILS_HEADER = (
    "タイトル,日次,レース日,レース場,"
    "レース回,レース名,距離(m),天候,風向,風速(m),波の高さ(cm),決まり手,"
    "単勝_艇番,単勝_払戻金,"
    "複勝_1着_艇番,複勝_1着_払戻金,複勝_2着_艇番,複勝_2着_払戻金,"
    "2連単_組番,2連単_払戻金,2連単_人気,"
    "2連複_組番,2連複_払戻金,2連複_人気,"
    "拡連複_1-2着_組番,拡連複_1-2着_払戻金,拡連複_1-2着_人気,"
    "拡連複_1-3着_組番,拡連複_1-3着_払戻金,拡連複_1-3着_人気,"
    "拡連複_2-3着_組番,拡連複_2-3着_払戻金,拡連複_2-3着_人気,"
    "3連単_組番,3連単_払戻金,3連単_人気,"
    "3連複_組番,3連複_払戻金,3連複_人気,"
    + ",".join(
        f"{k}着_{col}"
        for k in range(1, 7)
        for col in ["着順", "艇番", "登録番号", "選手名", "モーター番号", "ボート番号",
                    "展示タイム", "進入コース", "スタートタイミング", "レースタイム"]
    )
    + ",\n"
)


def _parse_details_file(text_file, csv_file):
    title = day = date = stadium = ""
    (result_win, result_place, result_exacta, result_quinella,
     result_qp, result_trifecta, result_trio, result_racer) = [""] * 8

    for line in text_file:
        if re.search(r"競走成績", line):
            text_file.readline()
            title   = text_file.readline()[:-1].strip()
            text_file.readline()
            line    = text_file.readline()
            day     = line[3:7].replace(" ", "")
            date    = line[17:27].replace(" ", "0")
            stadium = line[62:65].replace("　", "")

        if re.search(r"R", line) and re.search(r"H", line):
            if re.search(r"進入固定", line):
                line = line.replace("進入固定       H", "進入固定           H")

            race_round       = line[2:5].replace(" ", "0")
            race_name        = line[12:31].replace("　", "")
            distance         = line[36:40]
            weather          = line[43:45].strip()
            wind_direction   = line[50:52].strip()
            wind_velocity    = line[53:55].strip()
            wave_height      = line[60:63].strip()

            line              = text_file.readline()
            winning_technique = line[50:55].strip()
            text_file.readline()

            result_racer = ""
            line = text_file.readline()
            while line != "\n":
                result_racer += (
                    "," + line[2:4]
                    + "," + line[6]
                    + "," + line[8:12]
                    + "," + line[13:21]
                    + "," + line[22:24]
                    + "," + line[27:29]
                    + "," + line[30:35].strip()
                    + "," + line[38]
                    + "," + line[43:47]
                    + "," + line[52:58]
                )
                line = text_file.readline()

            line = text_file.readline()
            while line != "\n":
                if re.search(r"単勝", line):
                    if re.search(r"特払い", line):
                        line = line.replace("        特払い   ", "   特払い        ")
                    result_win = line[15] + "," + line[22:29].strip()

                if re.search(r"複勝", line):
                    if re.search(r"特払い", line):
                        line = line.replace("        特払い   ", "   特払い        ")
                    if len(line) <= 33:
                        result_place = line[15] + "," + line[22:29].strip() + ",,"
                    else:
                        result_place = (
                            line[15] + "," + line[22:29].strip()
                            + "," + line[31] + "," + line[38:45].strip()
                        )

                if re.search(r"２連単", line):
                    result_exacta = (
                        line[14:17] + "," + line[21:28].strip() + "," + line[36:38].strip()
                    )

                if re.search(r"２連複", line):
                    result_quinella = (
                        line[14:17] + "," + line[21:28].strip() + "," + line[36:38].strip()
                    )

                if re.search(r"拡連複", line):
                    result_qp = (
                        line[14:17] + "," + line[21:28].strip() + "," + line[36:38].strip()
                    )
                    line = text_file.readline()
                    result_qp += (
                        "," + line[17:20] + "," + line[24:31].strip() + "," + line[39:41].strip()
                    )
                    line = text_file.readline()
                    result_qp += (
                        "," + line[17:20] + "," + line[24:31].strip() + "," + line[39:41].strip()
                    )

                if re.search(r"３連単", line):
                    result_trifecta = (
                        line[14:19] + "," + line[21:28].strip() + "," + line[35:38].strip()
                    )

                if re.search(r"３連複", line):
                    result_trio = (
                        line[14:19] + "," + line[21:28].strip() + "," + line[35:38].strip()
                    )

                line = text_file.readline()

            csv_file.write(
                f"{title},{day},{date},{stadium},"
                f"{race_round},{race_name},{distance},"
                f"{weather},{wind_direction},{wind_velocity},{wave_height},"
                f"{winning_technique},"
                f"{result_win},{result_place},{result_exacta},{result_quinella},"
                f"{result_qp},{result_trifecta},{result_trio}"
                + result_racer + "\n"
            )


def parse_details(txt_dir: str = "downloads/results/txt/",
                  csv_path: str = "downloads/results/details/details.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="shift_jis") as f:
        f.write(_DETAILS_HEADER)

    txt_files = sorted(f for f in os.listdir(txt_dir) if re.search(r"\.TXT$", f, re.IGNORECASE))
    print(f"{len(txt_files)} 個のテキストファイルを処理します")

    for name in txt_files:
        with open(txt_dir + name, "r", encoding="shift_jis") as tf, \
             open(csv_path, "a", encoding="shift_jis") as cf:
            _parse_details_file(tf, cf)

    print(f"  → {csv_path}")


# ========================================================
# 払戻金のみ TXT → CSV（簡易版）
# ========================================================

_RESULTS_HEADER = (
    "タイトル,日次,レース日,レース場,レース回,"
    "3連単_組番,3連単_払戻金,3連複_組番,3連複_払戻金,"
    "2連単_組番,2連単_払戻金,2連複_組番,2連複_払戻金\n"
)


def _parse_results_file(text_file, csv_file):
    title = day = date = stadium = ""
    for contents in text_file:
        if re.search(r"競走成績", contents):
            text_file.readline()
            title   = text_file.readline()[:-1].strip()
            text_file.readline()
            line    = text_file.readline()
            day     = line[3:7].replace(" ", "")
            date    = line[17:27].replace(" ", "")
            stadium = line[62:65].replace("　", "")

        if re.search(r"払戻金", contents):
            line = text_file.readline()
            while line != "\n":
                results = (
                    line[10:13].strip() + "," + line[15:20]
                    + "," + line[21:28].strip() + "," + line[32:37]
                    + "," + line[38:45].strip() + "," + line[49:52]
                    + "," + line[53:60].strip() + "," + line[64:67]
                    + "," + line[68:75].strip()
                )
                csv_file.write(
                    f"{title},{day},{date},{stadium},{results}\n"
                )
                line = text_file.readline()


def parse_results(txt_dir: str = "downloads/results/txt/",
                  csv_path: str = "downloads/results/csv/results.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="shift_jis") as f:
        f.write(_RESULTS_HEADER)

    txt_files = sorted(f for f in os.listdir(txt_dir) if re.search(r"\.TXT$", f, re.IGNORECASE))
    print(f"{len(txt_files)} 個のテキストファイルを処理します")

    for name in txt_files:
        with open(txt_dir + name, "r", encoding="shift_jis") as tf, \
             open(csv_path, "a", encoding="shift_jis") as cf:
            _parse_results_file(tf, cf)

    print(f"  → {csv_path}")


# ========================================================
# CLI エントリーポイント
# ========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TXT → CSV 変換")
    parser.add_argument("target", choices=["timetable", "details", "results"],
                        help="変換対象")
    parser.add_argument("--out", default=None, help="出力 CSV パス（省略時はデフォルト）")
    args = parser.parse_args()

    if args.target == "timetable":
        kw = {"csv_path": args.out} if args.out else {}
        parse_timetable(**kw)
    elif args.target == "details":
        kw = {"csv_path": args.out} if args.out else {}
        parse_details(**kw)
    elif args.target == "results":
        kw = {"csv_path": args.out} if args.out else {}
        parse_results(**kw)

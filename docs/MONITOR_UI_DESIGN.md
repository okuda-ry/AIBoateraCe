# 自動モニタリング Web UI 設計書

## 概要

`run_auto.py` によるドライラン結果（`data/auto.db`）をブラウザで確認できる
ダッシュボードを Flask で構築する。

既存の予測 Web アプリ（`app.py`）と**同一プロセス・同一ポート**で動かす。
ナビバーにタブを追加するだけで済むため、最小限の変更で実現できる。

---

## アーキテクチャ

```
ブラウザ
  │
  ├── / ～ /predict      既存 Flask (app.py)   ← 変更なし
  │
  └── /monitor/*         新規ルート (monitor.py) ← 今回追加
                              │
                              └── data/auto.db  読み取り専用
                                  (recorder.py の関数を流用)
```

スケジューラ（`run_auto.py`）は**別プロセス**のまま。
Web UI は DB を読み取るだけなので、スケジューラが動いていなくても過去データを閲覧できる。

---

## ページ一覧

### 1. ダッシュボード `GET /monitor`

**目的:** 今日の損益を一目で確認する

表示内容:
- 日付セレクタ（デフォルト: 今日）
- サマリーカード × 4
  - スケジュール済み / 予測済み / ベット / 見送り レース数
- 損益カード
  - 総賭け金 / 総払戻 / 損益 / 回収率
- 戦略比較カード/テーブル（登録済み戦略を動的表示）
  - 賭けレース数 / 賭け金 / 回収 / 損益 / ROI / 的中数
- 当日レース一覧テーブル（後述 #2 へのリンク）

---

### 2. レース一覧 `GET /monitor/races?hd=YYYYMMDD`

**目的:** 当日の全レースの状態を一覧で把握する

表示内容（1行 = 1レース）:

| 列 | 内容 |
|----|------|
| 会場 / R | 平和島 3R |
| 発走 | 10:30 |
| 状態 | 予測済み / 未予測 / 結果確定 |
| 結果 | 3-1-5 / 未確定 |
| 払戻 | 3,530円 |
| 戦略別賭け金 | kelly / ip / strict_flat / true_kelly_cap / dutch_value / ip_conservative など |
| 戦略別損益 | 各戦略の損益（未購入は —） |

行クリック → レース詳細（#3）へ

戦略列は `kelly` / `ip` に固定しない。  
`models.strategies.STRATEGIES` に登録済みの戦略を基本表示し、DB に未知の戦略名が保存されている場合も後方に追加表示する。
横幅が長くなるため、戦略ごとに色付きバッジ・薄い背景色・区切り線で列グループを識別できるようにする。

---

### 3. レース詳細 `GET /monitor/race/<race_id>`

**目的:** 1レースのベット内訳を戦略ごとに詳しく確認する

表示内容:
- レースヘッダー（会場・日時・発走時刻）
- 結果バナー（結果確定時のみ）: 「3-1-5 払戻 3,530円」
- 戦略別ベットテーブル（タブ/セクション切り替え: 登録済み戦略を動的表示）
  - 組番 / 賭け金 / オッズ / EV / ステータス（win / lose / pending） / 払戻
- 戦略比較サマリー（このレース限定）

---

### 4. 累積履歴 `GET /monitor/history`

**目的:** 複数日にわたる損益トレンドを確認する

表示内容:
- 期間セレクタ（直近7日 / 30日 / 全期間）
- 日別損益グラフ（Chart.js 折れ線）
  - 戦略ごとの ROI を重ねて表示
- 累積損益グラフ（戦略別）
- 日別サマリーテーブル

---

## ファイル構成（追加・変更分のみ）

```
AiBoateraCe/
├── app.py                      # ナビバーにモニターリンクを追加
├── monitor.py                  # 新規: モニタリング UI の Flask Blueprint
│
├── templates/
│   ├── base.html               # ナビバーにモニタータブを追加
│   └── monitor/
│       ├── dashboard.html      # /monitor
│       ├── races.html          # /monitor/races
│       ├── race_detail.html    # /monitor/race/<race_id>
│       └── history.html        # /monitor/history
│
└── static/css/
    └── style.css               # モニター用スタイルを追記（既存に追加）
```

---

## データフロー

```
/monitor
  └── recorder.daily_summary(hd)       → サマリーカード
  └── recorder.strategy_summary(hd)    → 戦略比較カード/テーブル
  └── DB直接クエリ                      → レース一覧テーブル

/monitor/races
  └── races + bets を DB から集計
  └── models.strategies.STRATEGIES      → 登録済み戦略名
  └── strategy × race の賭金/損益表を動的生成

/monitor/race/<race_id>
  └── DB直接クエリ                      → レース詳細 + ベット内訳

/monitor/history
  └── strategy × date の日別損益/ROIを動的集計
  └── DB直接クエリ（hd GROUP BY）       → 日別集計
```

`recorder.py` の既存関数を最大限流用し、追加クエリは `monitor.py` に書く。

---

## 実装方針

### Blueprint を使う

`monitor.py` を Flask Blueprint として実装し、`app.py` に `register_blueprint` で追加する。
既存コードへの影響を最小化できる。

```python
# monitor.py
from flask import Blueprint
monitor_bp = Blueprint("monitor", __name__, url_prefix="/monitor")

# app.py に追加する行（2行だけ）
from monitor import monitor_bp
app.register_blueprint(monitor_bp)
```

### DB は読み取り専用

モニタリング UI から DB を書き換えることはしない。
スケジューラとの競合リスクをゼロにする。

### グラフは Chart.js（CDN）

追加ライブラリなし。既存の Bootstrap + Bootstrap Icons に加えて
Chart.js を CDN で読み込む。

### 自動リロード

ダッシュボードとレース一覧は `<meta http-equiv="refresh" content="60">` で
1分おきに自動更新する。レース当日に進捗をリアルタイムで確認できる。

---

## 実装ステップ

| # | 作業 | 工数目安 |
|---|------|---------|
| 1 | `monitor.py` Blueprint + ルート4本のスタブ | 小 |
| 2 | ダッシュボード (`/monitor`) + `dashboard.html` | 中 |
| 3 | レース一覧 (`/monitor/races`) + `races.html` | 中 |
| 4 | レース詳細 (`/monitor/race/<id>`) + `race_detail.html` | 中 |
| 5 | 累積履歴 (`/monitor/history`) + `history.html` + Chart.js | 大 |
| 6 | `base.html` にナビタブ追加 / `app.py` に Blueprint 登録 | 小 |

Step 1〜4 が最低限。Step 5（グラフ）は後回し可能。

---

## 画面遷移図

```
ナビバー
  ├── [予測] → / (既存)
  └── [モニター] → /monitor (ダッシュボード)
                      │
                      ├── [日付変更] → /monitor?hd=YYYYMMDD
                      ├── [レース一覧] → /monitor/races?hd=YYYYMMDD
                      │                    └── [行クリック] → /monitor/race/<id>
                      └── [累積履歴] → /monitor/history
```

---

## 注意事項

- `auto.db` が存在しない場合（ドライラン未実行）は「データなし」と表示するだけで
  エラーにしない
- スケジューラ（`run_auto.py`）の起動・停止はブラウザから行わない
  （プロセス管理の複雑さを避けるため）

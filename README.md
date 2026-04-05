# 競艇 AI 予測システム

過去の競艇データから **3連単（1〜3着の艇番と着順）** を予測し、
**リアルタイムオッズを取得して期待値プラスの組み合わせにのみ賭ける**
バリューベッティングまで行うシステム。

---

## 機能概要

| 機能 | 説明 |
|------|------|
| **モデル学習** | LightGBM LambdaRank で 3連単ランキングを学習 |
| **確率校正** | Isotonic Regression で softmax を真の当選率に補正 |
| **Web アプリ** | Flask でブラウザから出走表 URL を入力するだけで予測 |
| **リアルタイムスクレイピング** | boatrace.jp から出走表・直前情報・3連単オッズを自動取得 |
| **バリューベット** | 上位10予想の中で EV > 閾値 の組のみ 1/4 Kelly で購入 |
| **比例配分（フォールバック）** | オッズ未取得時は確率比例で N 円を配分 |
| **自動モニタリング** | 当日全レースを自動スケジューリング・予測・損益計算（ドライラン） |
| **LINE 通知** | 毎日 21:00 に日次損益レポートを LINE で受信 |
| **Optuna 最適化** | NDCG@3 を指標にハイパーパラメータを自動探索 |

---

## フォルダ構成

```
AiBoateraCe/
├── app.py                  # Flask Web アプリ（起動エントリーポイント）
├── train.py                # 学習エントリーポイント
├── predict.py              # CLI 予測スクリプト
├── run_auto.py             # 自動モニタリング起動スクリプト
├── requirements.txt
├── .env                    # LINE 認証情報（git 管理外・要作成）
├── .env.example            # .env のテンプレート
│
├── auto/                   # 自動モニタリングモジュール
│   ├── orchestrator.py     # スケジューラ（APScheduler）・ジョブ管理
│   ├── recorder.py         # SQLite による予測・結果・損益の記録
│   └── notifier.py         # LINE Messaging API 通知
│
├── data/                   # データパイプライン
│   ├── download.py         # LZH ダウンロード
│   ├── extract.py          # LZH → TXT 解凍
│   ├── parse.py            # TXT → CSV 変換
│   ├── preprocess.py       # 特徴量エンジニアリング・データ分割
│   ├── scraper.py          # boatrace.jp スクレイピング（スケジュール・オッズ・結果含む）
│   └── auto.db             # 損益記録 DB（自動生成・git 管理外）
│
├── models/                 # 機械学習モデル
│   ├── lgbm_ranking.py     # LightGBM LambdaRank（推奨）+ Optuna チューニング
│   ├── calibration.py      # 確率校正（Isotonic Regression）
│   ├── kelly_betting.py    # 掛け金最適化（バリューベット・Kelly・比例配分）
│   ├── ranking.py          # Self-Attention ランキングネット（TensorFlow）
│   ├── baseline.py         # 単勝ベースライン（TensorFlow）
│   └── saved/              # 学習済みモデル（train.py 実行後に生成・git 管理外）
│       ├── lgbm_booster.txt
│       ├── scalers.pkl
│       ├── race_columns.pkl
│       └── calibrator.pkl
│
├── templates/              # Jinja2 テンプレート
│   ├── base.html
│   ├── index.html
│   └── result.html
│
├── static/css/
│   └── style.css
│
├── notebooks/              # 分析用 Jupyter Notebook
│   └── dataframe.ipynb
│
├── docs/                   # 設計・戦略ドキュメント
│   ├── DESIGN.md           # 自動売買システム設計書
│   └── STRATEGY.md         # 収益化戦略書
│
└── downloads/              # データファイル（git 管理外）
    ├── racelists/
    │   ├── lzh/
    │   ├── txt/
    │   └── csv/
    └── results/
        ├── lzh/
        ├── txt/
        ├── csv/
        └── details/
```

---

## セットアップ

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

---

## データパイプライン

公式サイト（mbrace.or.jp）から取得したデータを段階的に変換する。
すでに `downloads/` 配下に CSV がある場合はスキップ可能。

### Step 1 — LZH ダウンロード

```bash
python data/download.py racelists --start 2009-09-01 --end 2024-09-01
python data/download.py results   --start 2009-09-01 --end 2024-09-01
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--start` | 2022-07-05 | 開始日 (YYYY-MM-DD) |
| `--end`   | 2024-09-01 | 終了日 (YYYY-MM-DD) |
| `--interval` | 1.0 | リクエスト間隔（秒）※ 1 秒以上推奨 |

### Step 2 — LZH → TXT 解凍

```bash
python data/extract.py racelists
python data/extract.py results
```

### Step 3 — TXT → CSV 変換

```bash
python data/parse.py timetable --out downloads/racelists/csv/timetable.csv
python data/parse.py details   --out downloads/results/details/details.csv
```

---

## モデル学習

```bash
# LightGBM LambdaRank（推奨）
python train.py lgbm

# Optuna でハイパーパラメータを最適化してから学習（時間がかかる）
python train.py lgbm --optuna --optuna-trials 100

# Self-Attention ランキングネット（TensorFlow）
python train.py ranking

# 単勝ベースライン
python train.py baseline
```

CSV パスをデフォルトから変更する場合:

```bash
python train.py lgbm \
    --timetable downloads/racelists/csv/timetable.csv \
    --details   downloads/results/details/details.csv
```

学習完了後、`models/saved/` に以下のファイルが保存される:

| ファイル | 内容 |
|---------|------|
| `lgbm_booster.txt` | LightGBM モデル本体 |
| `scalers.pkl` | 特徴量スケーラー |
| `race_columns.pkl` | レース特徴量の列名リスト |
| `calibrator.pkl` | 確率校正器（Isotonic Regression） |

---

## Web アプリ起動（推奨）

```bash
python app.py
```

ブラウザで **http://localhost:5000** を開く。

### 使い方

1. 出走表の URL をそのままペースト、または会場・日付・レース番号を選択
2. 予算（1 レースあたりの金額）を入力して「予測する」をクリック
3. 自動でスクレイピング → オッズ取得 → 予測 → バリューベット配分を表示

```
対応 URL 例:
  https://www.heiwajima.gr.jp/asp/heiwajima/kyogi/kyogihtml/index.htm?racenum=1
  https://www.boatrace.jp/owpc/pc/race/racelist?jcd=04&hd=20260320&rno=1
```

### 出力内容

| セクション | 内容 |
|-----------|------|
| レースヘッダー | 会場・日付・天候・風速・波高 |
| 信頼度バナー | 1着予測艇の校正済み確率（高/中/低） + 最有力組番 |
| 艇別スコア | スコア・1着確率バー・予測着順 |
| 3連単上位10 | Plackett-Luce 確率・オッズ・期待値（EV）一覧 |
| 推奨ベット | バリューベット（EV>5%の組のみ）または比例配分 |

---

## CLI 予測（上級者向け）

```bash
# URL から予測（オッズ自動取得）
python predict.py --url "https://www.boatrace.jp/owpc/pc/race/racelist?jcd=04&hd=20260320&rno=1"

# 会場・日付・レース番号で指定
python predict.py --venue 04 --date 20260320 --race 3 --budget 2000

# 最低期待値を変更（デフォルト 5%）
python predict.py --url "..." --min-edge 0.10

# オッズ取得をスキップして比例配分を使う
python predict.py --url "..." --no-odds

# デバッグ出力あり
python predict.py --url "..." --debug
```

---

## モデルの仕組み

### 予測パイプライン

```
出走表 + 直前情報 + オッズ
        ↓
  特徴量エンジニアリング (艇×レース)
        ↓
  LightGBM LambdaRank → スコア (6艇)
        ↓
  softmax → Isotonic Regression 校正 → 真の当選率 (6艇)
        ↓
  Plackett-Luce → 120通りの3連単確率
        ↓
  EV = 確率 × オッズ − 1  →  EV > 5% の組に 1/4 Kelly で配分
```

**NDCG@1, NDCG@3** を最適化することで「上位3艇の順序の質」を直接学習する。

### 損失関数（Plackett-Luce NLL）

TensorFlow モデル（ranking.py）で使用:

$$\mathcal{L} = -\sum_{k=1}^{3} \log \frac{e^{s_{\sigma(k)}}}{\sum_{j=k}^{6} e^{s_{\sigma(j)}}}$$

---

## 特徴量

### 艇レベル（6艇 × 24次元）

| 特徴量 | 次元 | 説明 |
|--------|------|------|
| 年齢・体重 | 2 | 選手プロフィール |
| 全国勝率・全国2連対率 | 2 | 全国成績 |
| 当地勝率・当地2連対率 | 2 | 開催場での成績 |
| モーター2連対率 | 1 | 使用モーターの成績 |
| ボート2連対率 | 1 | 使用ボートの成績 |
| 早見（展示タイム） | 1 | 展示レースのラップタイム |
| **今節平均ST** | **1** | **今節の平均スタートタイム（小さいほど早い＝有利）** |
| 今節成績 | 12 | 今節の各レース着順（最大12戦） |
| 級別エンコード | 1 | A1=3, A2=2, B1=1, B2=0 |
| **枠番** | **1** | **内側ほど有利（1=最内）** |

### レースレベル（~25次元）

風速・波高・日次・距離（数値）+ 天候・風向・レース場（one-hot）

### LightGBM 追加特徴量（相対特徴量）

数値10列について `(艇の値 - レース内平均)` を算出。
他艇と比べて有利・不利かを LightGBM が直接学べる。

---

## データ分割

時系列順に分割（ランダムだと未来データが訓練に混入するため）:

| セット | 割合 | 用途 |
|--------|------|------|
| Train  | 前 70% | 学習 |
| Val    | 次 10% | Early Stopping・確率校正 |
| Test   | 後 20% | 評価 |

---

## 評価指標

| 指標 | 意味 | 参考値 |
|------|------|--------|
| 3連単的中率（上位4通り）| 予測上位4通りのいずれかが正解 | ランダム ~3.3% |
| 3連単回収率 | 払戻合計 ÷ 賭け金合計 × 100 | 損益分岐 100% |
| NDCG@3 | 上位3艇の順序の質 | 最大 1.0 |
| Brier スコア | 確率校正の精度（校正前後で比較） | 小さいほど良い |
| 信頼度閾値別回収率 | 高信頼レースに絞った場合の回収率変化 | — |

---

## 掛け金最適化

| 方式 | オッズ | 説明 |
|------|--------|------|
| **バリューベット** | リアルタイムオッズ | EV > 最低期待値（デフォルト 5%）の組のみ 1/4 Kelly で購入（**推奨**） |
| **比例配分** | 不要 | 予測確率に比例して N 円を配分（オッズ未取得時のフォールバック） |
| Kelly（oracle） | 実際の払戻を使用 | バックテスト専用・性能上限の確認用 |

### EV（期待値）とは

**EV = Expected Value（期待値）** — 1 円賭けたときの期待リターン。

```
EV = モデル予測確率 × オッズ − 1
```

| 例 | モデル確率 | オッズ | EV | 判定 |
|----|-----------|--------|----|------|
| ① | 2% | 80倍 | 0.02 × 80 − 1 = **+0.6（+60%）** | 購入 ✅ |
| ② | 1% | 40倍 | 0.01 × 40 − 1 = **−0.6（−60%）** | 見送り ❌ |
| ③ | 5% | 10倍 | 0.05 × 10 − 1 = **−0.5（−50%）** | 見送り ❌ |

EV がプラスとは「モデルが市場（オッズ）より高く評価している」組み合わせ。
この優位性（エッジ）がある組み合わせにのみ賭けることで、テラ銭 25% を超える
リターンを理論上狙える。

### バリューベットについて

競艇のテラ銭は約 25%（期待収益率 75%）。全 120 通りを平均すると EV は必ず −25%。
「そこそこ当てる」だけでは赤字になるため、**EV プラスの組み合わせのみに絞る**ことが重要。

```
バリューベット = モデルが市場より高く評価している組み合わせだけを購入
```

#### 「期待値プラスが見つからない」と表示された場合

以下のいずれかの状況で表示される:

- 人気が集中してオッズが低く、全組み合わせで `モデル確率 × オッズ < 1` になっている
- 最低期待値（デフォルト 0.05 = 5%）が厳しすぎる → 0 に下げると購入対象が増える
- モデルの予測とオッズが一致しており、市場に対する優位性がない（見送りが最善）

> **ヒント**: `最低期待値` を `0` に設定すると、EV がわずかにプラスの組も購入対象になる。
> ただし優位性が小さいほど分散が大きくなるため、資金管理に注意が必要。

---

## 自動モニタリング（ドライラン）

当日の全レースを自動スケジューリングし、発走 **5 分前に予測**・発走 **35 分後に結果収集**を行う。
実際の投票は行わず、**「このレースで賭けていたら」という損益を記録する**。

### セットアップ

#### 1. APScheduler のインストール

```bash
pip install APScheduler>=3.10
```

#### 2. LINE 通知の設定（任意）

プロジェクトルートに `.env` ファイルを作成し、LINE の認証情報を記載する。

```bash
# .env.example をコピーして編集
cp .env.example .env
```

```
# .env
LINE_CHANNEL_ACCESS_TOKEN=<チャネルアクセストークン（長期）>
LINE_USER_ID=<自分のユーザーID (Uxxxxxxxx...)>
```

LINE の設定が不要な場合は `.env` を作成しなくても動作する（通知がスキップされるだけ）。

#### 3. 通知テスト

```bash
python auto/notifier.py
```

---

### 実行方法

```bash
# 今日の全レースを監視（ドライラン）
python run_auto.py

# 予算・最低EV を変更
python run_auto.py --budget 2000 --min-edge 0.03

# 日次サマリーをコンソールに表示して終了
python run_auto.py --summary

# DB の全レース結果を一覧表示
python run_auto.py --show-db

# 特定の日付を指定
python run_auto.py --date 20260405
```

### 動作フロー

```
起動時         scrape_schedule() で当日の全場・全レース時刻を取得
               → data/auto.db にスケジュールを保存

発走 5分前     予測実行（オッズ取得 → バリューベット計算）
               → DB に予測結果とベット候補を記録

発走 35分後    scrape_result() でレース結果を取得
               → DB に結果・払戻・損益を記録

毎日 21:00     日次損益レポートを LINE に送信
               （賭けたレース数・的中数・損益・回収率）

終了時（Ctrl+C）コンソールに日次サマリーを表示
```

### LINE 日次レポートの例

```
🚤 競艇 AI 日次レポート
📅 2026/04/05

▼ レース
  ベットあり : 8 レース
  見送り     : 42 レース
  的中       : 1 回  (12.5%)

▼ 損益
  賭け金     : 6,400 円
  払戻       : 8,250 円
  損益       : +1,850 円  📈
  回収率     : 128.9 %

⏰ 21:00 集計
```

### DB の確認

```bash
# 全レースの予測・結果一覧
python run_auto.py --show-db

# 日次サマリー
python run_auto.py --summary
```

---

## 注意事項

- 予測結果は参考情報です。投票は自己責任で行ってください。
- スクレイピングはサーバーに負荷をかけないよう適切な間隔を空けてください。
- 直前情報（展示タイム・ST）はスタート展示後にアクセスすると取得できます。
- オッズは締切直前に変動するため、早すぎる取得は参考値として扱ってください。

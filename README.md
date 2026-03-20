# 競艇 AI 予測システム

過去の競艇データから **3連単（1〜3着の艇番と着順）** を予測し、
**ブラウザ上でリアルタイム予測・掛け金配分** まで行うシステム。

---

## 機能概要

| 機能 | 説明 |
|------|------|
| **モデル学習** | LightGBM LambdaRank で 3連単ランキングを学習 |
| **Web アプリ** | Flask でブラウザから出走表 URL を入力するだけで予測 |
| **リアルタイムスクレイピング** | boatrace.jp から出走表・直前情報を自動取得 |
| **掛け金最適化** | 予算 N 円を Plackett-Luce 確率に比例して配分 |
| **信頼度分析** | 信頼度閾値を変えた場合の回収率をバックテスト |

---

## フォルダ構成

```
AiBoateraCe/
├── app.py                  # Flask Web アプリ（起動エントリーポイント）
├── train.py                # 学習エントリーポイント
├── predict.py              # CLI 予測スクリプト
│
├── data/
│   ├── download.py         # LZH ダウンロード
│   ├── extract.py          # LZH → TXT 解凍
│   ├── parse.py            # TXT → CSV 変換
│   ├── preprocess.py       # 特徴量エンジニアリング・データ分割
│   └── scraper.py          # boatrace.jp リアルタイムスクレイピング
│
├── models/
│   ├── lgbm_ranking.py     # LightGBM LambdaRank（推奨）
│   ├── ranking.py          # Self-Attention ランキングネット（TensorFlow）
│   ├── baseline.py         # 単勝ベースライン（TensorFlow）
│   ├── kelly_betting.py    # 掛け金最適化（Kelly 規準・比例配分）
│   └── saved/              # 学習済みモデル（train.py 実行後に生成）
│       ├── lgbm_booster.txt
│       ├── scalers.pkl
│       └── race_columns.pkl
│
├── templates/              # Jinja2 テンプレート
│   ├── base.html
│   ├── index.html
│   └── result.html
│
├── static/css/
│   └── style.css
│
├── requirements.txt
└── downloads/              # データファイル（.gitignore 対象）
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

学習完了後、`models/saved/` にモデルファイルが保存される。

---

## Web アプリ起動（推奨）

```bash
python app.py
```

ブラウザで **http://localhost:5000** を開く。

### 使い方

1. 出走表の URL をそのままペースト、または会場・日付・レース番号を選択
2. 予算（1 レースあたりの金額）を入力して「予測する」をクリック
3. 自動でスクレイピング → 予測 → 掛け金配分を表示

```
対応 URL 例:
  https://www.heiwajima.gr.jp/asp/heiwajima/kyogi/kyogihtml/index.htm?racenum=1
  https://www.boatrace.jp/owpc/pc/race/racelist?jcd=04&hd=20260320&rno=1
```

### 出力内容

| セクション | 内容 |
|-----------|------|
| レースヘッダー | 会場・日付・天候・風速・波高 |
| 信頼度バナー | 1着予測艇の softmax 確率（高/中/低） + 最有力組番 |
| 艇別スコア | スコア・1着確率バー・予測着順 |
| 3連単上位10 | Plackett-Luce 確率付き一覧 |
| 推奨ベット | 予算の比例配分（組番 × 金額バー） |

---

## CLI 予測（上級者向け）

```bash
# URL から予測
python predict.py --url "https://www.heiwajima.gr.jp/.../index.htm?racenum=3"

# 会場・日付・レース番号で指定
python predict.py --venue 04 --date 20260320 --race 3 --budget 2000

# デバッグ出力あり
python predict.py --url "..." --debug
```

---

## モデルの仕組み

### LightGBM LambdaRank（メインモデル）

```
特徴量 (艇×レース) → 相対特徴量追加 → LightGBM LambdaRank → スコア(6艇)
                                                                    ↓
                                           Plackett-Luce 確率 → 120通りの3連単確率
                                                                    ↓
                                                    比例配分で予算 N 円を配分
```

**NDCG@1, NDCG@3** を最適化することで「上位3艇の順序の質」を直接学習する。

### 損失関数（Plackett-Luce NLL）

TensorFlow モデル（ranking.py）で使用:

$$\mathcal{L} = -\sum_{k=1}^{3} \log \frac{e^{s_{\sigma(k)}}}{\sum_{j=k}^{6} e^{s_{\sigma(j)}}}$$

---

## 特徴量

### 艇レベル（6艇 × 23次元）

| 特徴量 | 次元 | 説明 |
|--------|------|------|
| 年齢・体重 | 2 | 選手プロフィール |
| 全国勝率・全国2連対率 | 2 | 全国成績 |
| 当地勝率・当地2連対率 | 2 | 開催場での成績 |
| モーター2連対率 | 1 | 使用モーターの成績 |
| ボート2連対率 | 1 | 使用ボートの成績 |
| 早見（展示タイム） | 1 | 展示レースのタイム |
| 今節成績 | 12 | 今節の各レース着順（最大12戦） |
| 級別エンコード | 1 | A1=3, A2=2, B1=1, B2=0 |
| **枠番** | **1** | **内側ほど有利（1=最内）** |

### レースレベル（~25次元）

風速・波高・日次・距離（数値）+ 天候・風向・レース場（one-hot）

### LightGBM 追加特徴量（相対特徴量）

数値9列について `(艇の値 - レース内平均)` を算出。
他艇と比べて有利・不利かを LightGBM が直接学べる。

---

## データ分割

時系列順に分割（ランダムだと未来データが訓練に混入するため）:

| セット | 割合 | 用途 |
|--------|------|------|
| Train  | 前 70% | 学習 |
| Val    | 次 10% | Early Stopping |
| Test   | 後 20% | 評価 |

---

## 評価指標

| 指標 | 意味 | 参考値 |
|------|------|--------|
| 3連単的中率（上位4通り）| 予測上位4通りのいずれかが正解 | ランダム ~3.3% |
| 3連単回収率 | 払戻合計 ÷ 賭け金合計 × 100 | 損益分岐 100% |
| NDCG@3 | 上位3艇の順序の質 | 最大 1.0 |
| 信頼度閾値別回収率 | 高信頼レースに絞った場合の回収率変化 | — |

---

## 掛け金最適化

| 方式 | オッズ | 説明 |
|------|--------|------|
| **比例配分** | 不要 | 予測確率に比例して N 円を配分（実運用可） |
| Kelly（oracle） | 実際の払戻を使用 | バックテスト専用・性能上限の確認用 |
| Kelly（実運用） | 事前オッズ必要 | 締切前にオッズをスクレイピングして使用 |

---

## 注意事項

- 予測結果は参考情報です。投票は自己責任で行ってください。
- スクレイピングはサーバーに負荷をかけないよう適切な間隔を空けてください。
- 直前情報（展示タイム）はスタート展示後にアクセスすると取得できます。

# Discord 通知ボット 設計書

## 1. 概要

競艇 AI の予測結果・レース結果・日次サマリーを Discord に通知する。
既存の LINE 通知（`auto/notifier.py`）と並行して動作し、**同一の通知イベントを両方に送れる**設計とする。

---

## 2. 通知方式の選定

### Discord Webhook（採用）

| 項目 | 内容 |
|------|------|
| 方式 | Discord サーバーのチャンネルに Webhook URL を発行し、HTTP POST で送信 |
| ライブラリ | 標準ライブラリのみ（`urllib.request`）、追加インストール不要 |
| 設定コスト | Webhook URL を `.env` に1行追記するだけ |
| 制限 | 送信のみ（受信・コマンド不可）。通知目的には十分 |

> Bot Token 方式（discord.py）は受信・スラッシュコマンドが必要になったとき、将来的に追加可能。

---

## 3. Discord の設定手順

```
1. Discord サーバーで通知用チャンネルを作成（例: #競艇AI通知）
2. チャンネルの設定 → 連携サービス → ウェブフック → 新しいウェブフック
3. 名前・アイコンを設定（例: 競艇AI）
4. 「ウェブフックURLをコピー」
5. プロジェクトの .env に追記:
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxx/yyyy
```

---

## 4. 実装ファイル構成

```
auto/
└── notifier.py          既存（LINE）+ Discord 送信を追記
.env                     DISCORD_WEBHOOK_URL を追記
```

新しいファイルは作らず、**既存の `notifier.py` に Discord 送信関数を追加**する。
orchestrator.py 側の変更は最小限（呼び出し1行追加のみ）。

---

## 5. 通知イベント一覧

| イベント | タイミング | 送信先 | 内容 |
|----------|-----------|--------|------|
| **買い目通知** | 発走5分前（予測完了後） | Discord + LINE | 会場・R番・買い目一覧・EV・オッズ |
| **結果通知** | 発走35分後（結果確定後） | Discord のみ | 3連単結果・払戻・各戦略の的中/損益 |
| **日次レポート** | 毎日21:00 | Discord + LINE | 当日の全戦略サマリー |

> LINE は文字数制限・フォーマット制約があるため、結果通知は Discord のみ（Embed で見やすく表示）。

---

## 6. Discord メッセージフォーマット

Discord Webhook は **Embed（埋め込みカード）** を使うと視認性が高い。

### 6-1. 買い目通知（Embed）

```
🚤 競艇AI  ベット候補
━━━━━━━━━━━━━━━━━━━━━━
📍 平和島 3R  10:30発走
💰 予算: 1,000円  信頼度: 高

[ kelly ]
  1-2-3  300円  @15.2  EV +52%
  2-1-3  200円  @22.1  EV +34%

[ ip ]
  1-2-3  200円  @15.2  EV +52%
  1-3-2  200円  @18.5  EV +41%

合計: 900円（ドライラン）
```

色: 青（#58a6ff）

### 6-2. 結果通知（Embed）

```
🏁 レース結果
━━━━━━━━━━━━━━━━━━━━━━
📍 平和島 3R
🎯 1-2-3  払戻 3,530円

[ kelly ] 的中 ✅  +10,290円
[ ip ]    外れ ❌    -700円
```

色: 的中あり=緑（#3fb950）/ 全外れ=赤（#f85149）/ 未ベット=グレー

### 6-3. 日次レポート（Embed）

```
📊 日次レポート  2026/04/10
━━━━━━━━━━━━━━━━━━━━━━
スケジュール: 120R  予測: 118R
ベット: 45R  見送り: 73R

戦略     賭け金    損益    ROI   的中
kelly   45,000  +3,200  107%  3回
ip      45,000  -1,500   97%  1回
```

色: 損益プラス=緑 / マイナス=赤

---

## 7. `notifier.py` への追記内容

### 追加する関数

```python
# Discord Webhook 送信コア
def send_discord(embeds: list[dict]) -> bool

# 買い目通知（Discord）
def notify_discord_bet(race: dict, strategy_bets: dict, odds_dict: dict, confidence: float) -> bool

# 結果通知（Discord）
def notify_discord_result(race: dict, result_combo: str, payout: int, strategy_bets: dict) -> bool

# 日次レポート（Discord）
def notify_discord_daily(summary: dict, strategies: list[dict]) -> bool
```

### `.env` 追加キー

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxx/yyyy
```

---

## 8. `orchestrator.py` への変更内容

変更は最小限。既存の `job_predict` / `job_collect_result` / `job_line_daily_report` に1〜2行追加するだけ。

```python
# job_predict の末尾に追加
from auto.notifier import notify_discord_bet
notify_discord_bet(race, strategy_bets, odds_dict, confidence)

# job_collect_result の末尾に追加
from auto.notifier import notify_discord_result
notify_discord_result(race, combo, payout, strategy_bets)

# job_line_daily_report の末尾に追加（戦略別情報も渡す）
from auto.notifier import notify_discord_daily
notify_discord_daily(s, strategy_summary(hd))
```

---

## 9. 動作フロー（全体）

```
run_auto.py 起動
    │
    ├─ 発走5分前
    │    job_predict()
    │        └─ run_prediction() → run_all_strategies()
    │               └─ save_all_strategies()  ← DB保存
    │               └─ notify_discord_bet()   ← Discord通知 ★
    │               └─ notify_bet_signal()    ← LINE通知（既存）
    │
    ├─ 発走35分後
    │    job_collect_result()
    │        └─ scrape_result() → update_result()  ← DB更新
    │               └─ notify_discord_result()     ← Discord通知 ★
    │
    └─ 毎日21:00
         job_line_daily_report()
             └─ daily_summary() + strategy_summary()
                    └─ notify_discord_daily()   ← Discord通知 ★
                    └─ notify_daily_summary()   ← LINE通知（既存）
```

---

## 10. エラーハンドリング方針

- Discord 通知の失敗は **警告ログのみ** で処理を止めない（メイン処理に影響させない）
- Webhook URL が未設定の場合はスキップ（LINE と同じ挙動）
- HTTP 429（レートリミット）時は `retry-after` ヘッダーを見て1回リトライ

---

## 11. 実装ステップ

1. Discord サーバーで Webhook URL を発行し `.env` に追記
2. `auto/notifier.py` に Discord 関数3本を追加
3. `auto/orchestrator.py` に通知呼び出しを追加（3箇所）
4. `python auto/notifier.py` でテスト送信して確認
5. 既存テストに `test_notifier.py` を追加

---

## 12. 将来拡張（Bot Token 方式）

現在の Webhook 方式では「受信」ができない。将来的に以下が必要になった場合は discord.py に移行する。

| 機能 | 説明 |
|------|------|
| `/今日の買い目` | スラッシュコマンドで当日のベット一覧を返す |
| `/結果 平和島 3R` | 指定レースの結果を返す |
| リアクション集計 | ベット候補に ✅/❌ でフィードバック収集 |

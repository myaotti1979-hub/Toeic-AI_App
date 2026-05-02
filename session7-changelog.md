# TOEIC AI練習アプリ — セッション7 変更履歴
## v2026.05.02b → v2026.05.02d
## 日付: 2026-05-02 (JST)

---

## Part 1: コードレビュー改修 (v2026.05.02c)

### セキュリティ
- `safeImgUrl()`: imgUrl XSS遮断（レンダリング2箇所+インポート3箇所）
- `ctxBlock` label `esc()`: talk_type経由XSS遮断
- `w.level`/`q.target.pos` フォールバック `esc()`: vocab画面XSS遮断

### バグ修正
- 模試タイマー `setTimeout` leak → `setInterval` + 自動クリーンアップ

### パフォーマンス
- `judgeGetRecentByParts()`: HOME画面IRT推定を全件→パート別直近50件に

### コード品質
- `MASTERY_STREAK`/`PART_RECENT_N`/`BATCH_ERROR_LIMIT`/`DEFAULT_DAILY_GOAL` 定数化
- `localDateStr()`/`localYesterdayStr()` ヘルパー追加
- 空 `catch(()=>{})` → `console.warn` 付きに7箇所修正
- `render()` 冒頭にタイマークリーンアップ

---

## Part 2: 900点突破 + 英語力向上機能 (v2026.05.02d)

### ⚡ Part 5 速度ドリル
- タイムアタック時に10秒制限モード（本番目標20秒の半分で訓練）

### 🔍 間違い理由の記録・分析
- 📚知識不足/👁️読み飛ばし/⏱️時間切れ/🪤引っかけの4パターン記録
- 実力診断に間違いパターン分析グラフ追加

### 🎙️ シャドーイング練習
- Part 3/4回答後に1文ずつWeb Speech再生（データ追加ゼロ）
- テキストブラー→表示切替、前/次ナビ、全文通し再生

### 📖 WPM計測
- Part 7の読速を自動計測（目標: 150 WPM）

### 📎 コロケーションクイズ
- 例文中の単語を空欄化、同品詞の選択肢から正解を選ぶ

### 🔍 不正解選択肢の品質チェック（Streamlit）
- Geminiで不正解のもっともらしさをA/B/C 3段階評価

### ✍️ 英作文練習
- Part 5回答後に同じ文法で英作文→Gemini添削

---

## 出力ファイル
- `index.html` — v2026.05.02d (7,891行)
- `toeic_generator_app.py` — v2026.05.02d (3,910行)
- `sw.js` — v2026.05.02d

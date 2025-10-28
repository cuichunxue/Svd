# クイックスタートガイド

## インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt
```

## 5分で始める異常検知

### ステップ1: ライブラリのインポート

```python
import numpy as np
from src.svd_anomaly_detector import SVDAnomalyDetector
```

### ステップ2: データの準備

```python
# サンプルデータの生成（正弦波 + ノイズ）
t = np.linspace(0, 10*np.pi, 1000)
data = np.sin(t) + 0.1 * np.random.randn(1000)

# 異常を注入
data[500] = 5.0  # スパイク異常
```

### ステップ3: 検出器の作成と学習

```python
# 検出器の初期化
detector = SVDAnomalyDetector(
    window_size=20,
    threshold=2.5
)

# 正常データで学習（最初の70%を使用）
detector.fit(data[:700])
```

### ステップ4: 異常検出

```python
# 異常を検出
anomalies = detector.predict(data)

# 異常が検出された位置
anomaly_indices = np.where(anomalies == 1)[0]
print(f"異常検出位置: {anomaly_indices}")
```

### ステップ5: 結果の確認

```python
# 異常スコアの取得
scores = detector.get_anomaly_scores(data)

# モデル情報
print(f"使用ランク: {detector.get_optimal_rank()}")
print(f"検出された異常数: {np.sum(anomalies)}")
```

## サンプルプログラムの実行

### 基本例

```bash
python examples/basic_example.py
```

合成データを使った完全な異常検知の例を実行します。
結果は `examples/anomaly_detection_results.png` に保存されます。

### 高度な例（多変量）

```bash
python examples/advanced_example.py
```

複数の時系列（CPU、メモリ、ネットワーク）の異常検知を実演します。
結果は `examples/multivariate_anomaly_detection.png` に保存されます。

## よくある使用パターン

### パターン1: 単純な異常検出

```python
from src.svd_anomaly_detector import detect_anomalies_svd

# データの準備
data = your_time_series_data

# 異常検出（ワンライナー）
anomalies, scores = detect_anomalies_svd(data, window_size=15)
```

### パターン2: パラメータのチューニング

```python
detector = SVDAnomalyDetector(
    window_size=30,     # より長いパターンを検出
    rank=5,             # 5つの主成分を使用
    threshold=2.0       # より敏感な検出
)
```

### パターン3: リアルタイム監視

```python
# 初期学習
detector.fit(historical_normal_data)

# 新しいデータが来るたびに検出
while True:
    new_data = get_new_data()
    if len(new_data) >= detector.window_size:
        anomalies = detector.predict(new_data)
        if np.any(anomalies):
            alert("異常を検出しました！")
```

## トラブルシューティング

### 問題: 異常が検出されない

**解決策**: 閾値を下げてみる

```python
detector = SVDAnomalyDetector(window_size=20, threshold=2.0)  # デフォルト3.0から下げる
```

### 問題: 誤検知が多い

**解決策**: 閾値を上げる、またはウィンドウサイズを調整

```python
detector = SVDAnomalyDetector(window_size=30, threshold=3.5)
```

### 問題: メモリエラー

**解決策**: データを分割処理するか、ウィンドウサイズを小さくする

```python
# データを小さなチャンクで処理
chunk_size = 5000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    anomalies = detector.predict(chunk)
```

## 次のステップ

- `README.md` で理論的背景を学ぶ
- `examples/` のコードを読んで詳細な使用法を理解する
- 自分のデータで実験してパラメータを調整する
- 多変量データの処理方法を学ぶ

## ヘルプが必要な場合

問題が発生した場合は、GitHubのIssuesで質問してください。

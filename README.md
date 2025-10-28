# SVD-based Time Series Anomaly Detection

時系列データの異常検知を特異値分解（Singular Value Decomposition, SVD）を用いて実装したPythonライブラリです。

## 概要

このライブラリは、SVDを使用して時系列データの異常を検出します。スライディングウィンドウを使用してトラジェクトリ行列を作成し、SVDで次元削減を行い、再構成誤差に基づいて異常を検出します。

### 主な特徴

- スライディングウィンドウベースのトラジェクトリ行列構築
- 自動ランク選択（説明分散比率に基づく）
- 再構成誤差による異常スコアリング
- 単変量・多変量時系列データ両対応
- シンプルで直感的なAPI

## 理論背景

### SVDによる異常検知の原理

1. **トラジェクトリ行列の作成**: 時系列データからスライディングウィンドウを使用してハンケル行列を構築
2. **SVD分解**: トラジェクトリ行列を特異値分解し、主要成分を抽出
3. **再構成**: 上位k個の特異値を使用してデータを再構成
4. **異常検出**: 元のデータと再構成データの差（再構成誤差）が閾値を超える場合に異常と判定

正常なデータのパターンは少数の主要成分で表現できますが、異常なデータは大きな再構成誤差を生じます。

## インストール

### 必要な依存関係

```bash
pip install -r requirements.txt
```

### 手動インストール

```bash
pip install numpy scipy matplotlib
```

## 使用方法

### 基本的な使い方

```python
import numpy as np
from src.svd_anomaly_detector import SVDAnomalyDetector

# 時系列データの準備
time_series = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000) * 0.1

# 検出器の初期化と学習
detector = SVDAnomalyDetector(
    window_size=20,      # ウィンドウサイズ
    rank=None,           # 自動決定
    threshold=2.5        # 異常判定の閾値倍数
)

# 正常データで学習
detector.fit(time_series[:700])

# 異常検出
anomalies = detector.predict(time_series)
anomaly_scores = detector.get_anomaly_scores(time_series)

print(f"検出された異常: {np.sum(anomalies)}個")
print(f"使用ランク: {detector.get_optimal_rank()}")
```

### 簡易関数の使用

```python
from src.svd_anomaly_detector import detect_anomalies_svd

# ワンライナーで異常検出
anomalies, scores = detect_anomalies_svd(
    time_series,
    window_size=10,
    threshold=3.0
)
```

## 実行例

### 基本例（合成データ）

```bash
python examples/basic_example.py
```

このスクリプトは以下を実行します：
- 正弦波ベースの合成時系列データ生成
- ランダムな異常の注入
- SVD検出器の学習と異常検出
- 精度評価（Precision, Recall, F1スコア）
- 結果の可視化

出力例：
```
SVD-based Time Series Anomaly Detection Example
============================================================

1. Generating synthetic time series data...
   Total points: 1000
   True anomalies: 50

2. Training SVD anomaly detector...
   Window size: 20
   Optimal rank: 5
   Explained variance (top 5): [0.421 0.289 0.156 0.078 0.034]

3. Detecting anomalies...
   Detected anomalies: 127

4. Results:
   Precision: 0.378
   Recall: 0.960
   F1 Score: 0.542
```

### 高度な例（多変量データ）

```bash
python examples/advanced_example.py
```

システムメトリクス（CPU、メモリ、ネットワーク）を模擬した多変量時系列データでの異常検出を実演します。

## パラメータ調整

### window_size（ウィンドウサイズ）

- **小さい値（5-10）**: 局所的な異常に敏感、ノイズに反応しやすい
- **中程度（10-30）**: バランスの取れた検出
- **大きい値（30-100）**: 大域的なパターン変化を検出、計算コスト増

### rank（ランク）

- **None**: 自動選択（推奨）- 95%の分散を説明する成分数
- **低い値（1-3）**: 強い次元削減、明確な異常のみ検出
- **高い値**: より詳細なパターンを保持、偽陰性が減る

### threshold（閾値）

- **低い値（1.5-2.0）**: 高感度、多くの異常を検出（偽陽性増加の可能性）
- **中程度（2.0-3.0）**: 標準的な設定
- **高い値（3.0-4.0）**: 保守的、明確な異常のみ検出

## API リファレンス

### SVDAnomalyDetector クラス

#### メソッド

- `fit(time_series)`: 正常データで学習
- `predict(time_series)`: 異常を予測（バイナリ配列を返す）
- `get_anomaly_scores(time_series)`: 異常スコア（再構成誤差）を取得
- `get_optimal_rank()`: 使用されているランク数を取得
- `get_explained_variance_ratio()`: 各特異値の説明分散比を取得

#### パラメータ

- `window_size` (int): スライディングウィンドウのサイズ
- `rank` (int, optional): 保持する特異値の数（Noneで自動決定）
- `threshold` (float): 異常判定の閾値倍数（デフォルト: 3.0）

## アプリケーション例

- サーバーメトリクス監視
- 製造業の品質管理
- 金融時系列の異常取引検出
- IoTセンサーデータの異常検知
- ネットワークトラフィック分析

## 技術的詳細

### トラジェクトリ行列

長さNの時系列とウィンドウサイズLから、L×(N-L+1)のトラジェクトリ行列（ハンケル行列）を構築：

```
T = [x[0]   x[1]   x[2]   ... x[N-L]  ]
    [x[1]   x[2]   x[3]   ... x[N-L+1]]
    [x[2]   x[3]   x[4]   ... x[N-L+2]]
    [...    ...    ...    ... ...     ]
    [x[L-1] x[L]   x[L+1] ... x[N-1]  ]
```

### SVD分解

```
T = U Σ V^T
```

ここで：
- U: 左特異ベクトル（時間パターン）
- Σ: 特異値の対角行列（重要度）
- V^T: 右特異ベクトル（空間パターン）

### 再構成

上位k個の成分を使用して再構成：

```
T_k = U_k Σ_k V_k^T
```

### 異常スコア

各ウィンドウの再構成誤差：

```
error = ||T - T_k||_F
```

## 制限事項

- ウィンドウサイズは時系列長より小さい必要があります
- 学習データは正常なデータのみを含むべきです
- 非常に長い時系列では計算コストが高くなる可能性があります
- 突発的な異常よりも段階的な変化の方が検出しやすい傾向があります

## ライセンス

MIT License

## 参考文献

1. Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for Time Series. Springer.
2. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3), 1-58.

## 貢献

プルリクエストや問題報告を歓迎します。

## サポート

問題が発生した場合は、GitHubのIssuesセクションで報告してください。

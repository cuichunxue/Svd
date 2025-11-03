# Plotlyインタラクティブ可視化ガイド

このディレクトリには、SVD異常検知システムのためのインタラクティブなPlotly可視化が含まれています。

## 📊 生成される可視化

### 1. 単一シナリオ可視化 (`plotly_single_scenario.html`)

**特徴**:
- 時系列データと検出された異常を同時表示
- 異常スコア（再構成誤差）のプロット
- 閾値ラインの表示
- インタラクティブなズーム・パン機能

**表示内容**:
- **上部パネル**: 時系列データ（青線）と検出された異常（赤×マーク）
- **下部パネル**: 異常スコア（緑の塗りつぶし）と閾値（赤点線）

### 2. 複数シナリオ比較 (`plotly_multiple_scenarios.html`)

**特徴**:
- 3つの異なる異常タイプを同時比較
  - スパイク異常
  - 段階的変化
  - ノイズバースト
- 各シナリオの時系列と異常スコアを並列表示

**用途**:
- 異なる異常タイプでの検出性能の比較
- パラメータ調整の効果確認
- プレゼンテーション資料

### 3. 3D異常スコアランドスケープ (`plotly_3d_landscape.html`)

**特徴**:
- ウィンドウサイズと時間インデックスに対する異常スコアの3D表示
- 回転・ズーム可能な3Dビュー
- パラメータ感度の視覚的理解

**表示内容**:
- **X軸**: ウィンドウインデックス（時間）
- **Y軸**: ウィンドウサイズ（5～40）
- **Z軸**: 異常スコア
- **色**: 異常スコアの大きさ（Viridisカラースケール）

## 🚀 使い方

### 基本的な実行

```bash
# 依存関係のインストール
pip install plotly

# デモの実行（3つの可視化を生成）
python examples/plotly_visualization.py
```

### カスタム可視化の作成

```python
import numpy as np
from src.svd_anomaly_detector import SVDAnomalyDetector
from examples.plotly_visualization import create_interactive_anomaly_plot

# データの準備
time_series = np.sin(np.linspace(0, 10*np.pi, 1000))
time_series[500] = 5.0  # 異常を注入

# 検出器の学習
detector = SVDAnomalyDetector(window_size=20, threshold=2.5)
detector.fit(time_series[:600])

# インタラクティブプロットの作成
fig = create_interactive_anomaly_plot(
    time_series,
    detector,
    title="My Custom Analysis"
)

# HTMLとして保存
fig.write_html("my_analysis.html")

# またはJupyterで直接表示
fig.show()
```

## 🎯 インタラクティブ機能

生成されたHTMLファイルをブラウザで開くと、以下の機能が使えます：

### 基本操作
- **ズームイン**: ドラッグで範囲選択
- **ズームアウト**: ダブルクリック
- **パン**: Shift + ドラッグ
- **ホバー情報**: データポイントにマウスを合わせると詳細表示

### トレース制御
- **凡例クリック**: トレースの表示/非表示を切り替え
- **凡例ダブルクリック**: そのトレースのみ表示

### 3D操作（3Dランドスケープのみ）
- **回転**: ドラッグ
- **ズーム**: スクロール
- **パン**: 右クリック + ドラッグ

### ツールバー機能
- **📷 カメラアイコン**: PNGとして保存
- **🔍 ズームツール**: 範囲選択ズーム
- **⬅️ 矢印**: ズーム履歴を戻る/進む
- **🏠 ホームアイコン**: 初期表示にリセット

## 📈 可視化の活用例

### 1. 異常検知結果の説明
```python
# レポート用の可視化生成
from examples.plotly_visualization import create_interactive_anomaly_plot

fig = create_interactive_anomaly_plot(
    production_data,
    trained_detector,
    title="Production System Anomaly Detection - Week 45"
)
fig.write_html("weekly_report.html")
```

### 2. パラメータチューニング
```python
# 異なる閾値での比較
scenarios = []
for threshold in [1.5, 2.5, 3.5]:
    detector = SVDAnomalyDetector(window_size=20, threshold=threshold)
    detector.fit(train_data)
    scenarios.append({
        'name': f'Threshold={threshold}',
        'time_series': test_data,
        'anomalies': detector.predict(test_data),
        'scores': detector.get_anomaly_scores(test_data),
        'threshold': detector.mean + detector.threshold * detector.std
    })

from examples.plotly_visualization import create_multiple_scenarios_plot
fig = create_multiple_scenarios_plot(scenarios)
fig.write_html("threshold_comparison.html")
```

### 3. パラメータ感度分析
```python
# ウィンドウサイズの影響を3Dで可視化
from examples.plotly_visualization import create_3d_anomaly_landscape

fig = create_3d_anomaly_landscape(
    time_series,
    base_detector,
    window_range=(5, 50, 5)
)
fig.write_html("parameter_sensitivity.html")
```

## 🎨 カスタマイズオプション

### 色の変更
```python
fig.update_traces(
    line=dict(color='purple'),
    selector=dict(name='Time Series')
)
```

### レイアウトの調整
```python
fig.update_layout(
    height=1000,
    width=1600,
    template='plotly_dark',  # ダークテーマ
    font=dict(size=14, family='Courier New')
)
```

### 複数のトレースを追加
```python
# 真の異常値も表示
true_anomalies = np.array([...])
fig.add_trace(
    go.Scatter(
        x=np.where(true_anomalies)[0],
        y=time_series[true_anomalies == 1],
        mode='markers',
        name='True Anomalies',
        marker=dict(color='gold', size=10, symbol='star')
    ),
    row=1, col=1
)
```

## 💡 Tips

### パフォーマンス最適化
- 大規模データ（10,000点以上）の場合、サンプリングを検討
```python
# データをダウンサンプリング
step = 10
sampled_data = time_series[::step]
```

### Jupyter Notebookでの使用
```python
# Jupyterで直接表示
import plotly.io as pio
pio.renderers.default = 'notebook'
fig.show()
```

### 静的画像としてエクスポート
```python
# PNG画像として保存（kaleido が必要）
fig.write_image("anomaly_plot.png", width=1600, height=900)
```

## 🔧 トラブルシューティング

### Q: HTMLファイルが開けない
**A**: ブラウザのセキュリティ設定を確認してください。ローカルファイルの実行が許可されている必要があります。

### Q: 3D可視化が重い
**A**: ウィンドウサイズの範囲を狭めるか、ステップを大きくしてください：
```python
window_range=(10, 30, 10)  # より少ないサンプル点
```

### Q: ホバー情報が表示されない
**A**: `hovertemplate` を確認し、ブラウザの開発者ツールでエラーをチェックしてください。

### Q: グラフが表示されない
**A**: Plotlyのバージョンを確認してください：
```bash
pip install --upgrade plotly
```

## 📚 参考資料

- [Plotly公式ドキュメント](https://plotly.com/python/)
- [Plotly Graph Objects](https://plotly.com/python/graph-objects/)
- [Subplotsガイド](https://plotly.com/python/subplots/)
- [3D Surface Plots](https://plotly.com/python/3d-surface-plots/)

## 🤝 貢献

改善案やバグ報告は GitHubのIssuesでお願いします。

---

**Happy Visualizing! 📊✨**

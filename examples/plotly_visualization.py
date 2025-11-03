"""
Interactive Plotly Visualization for SVD Anomaly Detection

This script creates interactive visualizations using Plotly for:
- Time series data with anomaly highlights
- Anomaly scores (reconstruction errors)
- Multiple scenarios comparison
- Interactive exploration with zoom, pan, and hover information
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from svd_anomaly_detector import SVDAnomalyDetector


def create_interactive_anomaly_plot(time_series, detector, title="SVD Anomaly Detection"):
    """
    Create an interactive Plotly visualization of anomaly detection results.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data
    detector : SVDAnomalyDetector
        Trained detector
    title : str
        Plot title

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    # Get predictions and scores
    anomalies = detector.predict(time_series)
    scores = detector.get_anomaly_scores(time_series)

    # Create subplots: time series on top, anomaly scores on bottom
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Time Series with Detected Anomalies', 'Anomaly Scores (Reconstruction Error)'),
        row_heights=[0.6, 0.4]
    )

    # Plot 1: Time series
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(time_series)),
            y=time_series,
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=1),
            hovertemplate='<b>Index</b>: %{x}<br><b>Value</b>: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Highlight anomalies
    anomaly_indices = np.where(anomalies == 1)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomaly_indices,
                y=time_series[anomaly_indices],
                mode='markers',
                name='Detected Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='<b>Anomaly at</b>: %{x}<br><b>Value</b>: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Plot 2: Anomaly scores
    window_indices = np.arange(len(scores))
    threshold = detector.mean + detector.threshold * detector.std

    fig.add_trace(
        go.Scatter(
            x=window_indices,
            y=scores,
            mode='lines',
            name='Anomaly Score',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            hovertemplate='<b>Window</b>: %{x}<br><b>Score</b>: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[0, len(scores)-1],
            y=[threshold, threshold],
            mode='lines',
            name=f'Threshold ({threshold:.2f})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Threshold</b>: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Highlight areas above threshold
    above_threshold = scores > threshold
    if np.any(above_threshold):
        fig.add_trace(
            go.Scatter(
                x=window_indices[above_threshold],
                y=scores[above_threshold],
                mode='markers',
                name='Above Threshold',
                marker=dict(color='red', size=6),
                hovertemplate='<b>Window</b>: %{x}<br><b>Score</b>: %{y:.3f}<br><b>Status</b>: Anomaly<extra></extra>'
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_xaxes(title_text="Time Index", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Reconstruction Error", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333')
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig


def create_multiple_scenarios_plot(scenarios_data):
    """
    Create interactive comparison of multiple scenarios.

    Parameters
    ----------
    scenarios_data : list of dict
        Each dict contains 'name', 'time_series', 'anomalies', 'scores', 'threshold'

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    n_scenarios = len(scenarios_data)

    fig = make_subplots(
        rows=n_scenarios, cols=2,
        subplot_titles=[item for scenario in scenarios_data
                       for item in [f"{scenario['name']} - Time Series",
                                  f"{scenario['name']} - Anomaly Scores"]],
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
        column_widths=[0.6, 0.4]
    )

    for idx, scenario in enumerate(scenarios_data, 1):
        time_series = scenario['time_series']
        anomalies = scenario['anomalies']
        scores = scenario['scores']
        threshold = scenario['threshold']

        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(time_series)),
                y=time_series,
                mode='lines',
                name=f"{scenario['name']}",
                line=dict(width=1),
                showlegend=False,
                hovertemplate='<b>Value</b>: %{y:.3f}<extra></extra>'
            ),
            row=idx, col=1
        )

        # Anomaly points
        anomaly_indices = np.where(anomalies == 1)[0]
        if len(anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_indices,
                    y=time_series[anomaly_indices],
                    mode='markers',
                    marker=dict(color='red', size=5, symbol='x'),
                    showlegend=False,
                    hovertemplate='<b>Anomaly</b><extra></extra>'
                ),
                row=idx, col=1
            )

        # Anomaly scores
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(scores)),
                y=scores,
                mode='lines',
                line=dict(color='green', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)',
                showlegend=False,
                hovertemplate='<b>Score</b>: %{y:.3f}<extra></extra>'
            ),
            row=idx, col=2
        )

        # Threshold line
        fig.add_trace(
            go.Scatter(
                x=[0, len(scores)-1],
                y=[threshold, threshold],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False,
                hovertemplate=f'<b>Threshold</b>: {threshold:.3f}<extra></extra>'
            ),
            row=idx, col=2
        )

    fig.update_xaxes(title_text="Time Index", row=n_scenarios, col=1)
    fig.update_xaxes(title_text="Window Index", row=n_scenarios, col=2)

    fig.update_layout(
        title=dict(
            text="Multi-Scenario Anomaly Detection Comparison",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        height=300 * n_scenarios,
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_3d_anomaly_landscape(time_series, detector, window_range=(5, 50, 5)):
    """
    Create 3D visualization showing how anomaly scores vary with window size.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data
    detector : SVDAnomalyDetector
        Base detector (will be retrained with different window sizes)
    window_range : tuple
        (min_window, max_window, step)

    Returns
    -------
    plotly.graph_objects.Figure
        3D surface plot
    """
    min_w, max_w, step = window_range
    window_sizes = range(min_w, max_w + 1, step)

    # Calculate scores for different window sizes
    scores_matrix = []
    max_length = 0

    for ws in window_sizes:
        try:
            temp_detector = SVDAnomalyDetector(window_size=ws, threshold=detector.threshold)
            train_size = int(0.6 * len(time_series))
            temp_detector.fit(time_series[:train_size])
            scores = temp_detector.get_anomaly_scores(time_series)
            scores_matrix.append(scores)
            max_length = max(max_length, len(scores))
        except:
            continue

    # Pad scores to same length
    padded_scores = []
    for scores in scores_matrix:
        if len(scores) < max_length:
            padded = np.pad(scores, (0, max_length - len(scores)), constant_values=np.nan)
        else:
            padded = scores[:max_length]
        padded_scores.append(padded)

    Z = np.array(padded_scores)
    X = np.arange(max_length)  # Window index
    Y = np.array(list(window_sizes))  # Window sizes

    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title="Anomaly Score"),
        hovertemplate='<b>Window Index</b>: %{x}<br><b>Window Size</b>: %{y}<br><b>Score</b>: %{z:.3f}<extra></extra>'
    )])

    fig.update_layout(
        title='Anomaly Score Landscape: Window Size vs. Time',
        scene=dict(
            xaxis_title='Window Index',
            yaxis_title='Window Size',
            zaxis_title='Anomaly Score',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        template='plotly_white'
    )

    return fig


def demo_single_scenario():
    """Demo: Single scenario visualization."""
    print("Demo 1: Single Scenario Interactive Visualization")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    t = np.linspace(0, 10*np.pi, 1000)
    time_series = np.sin(t) + 0.5*np.sin(3*t) + 0.1*np.random.randn(1000)

    # Inject anomalies
    time_series[300] = 5.0  # Spike
    time_series[600] = -4.0  # Drop
    time_series[800:850] += 2.0  # Level shift

    # Train detector
    detector = SVDAnomalyDetector(window_size=20, threshold=2.5)
    train_size = 600
    detector.fit(time_series[:train_size])

    # Create visualization
    fig = create_interactive_anomaly_plot(
        time_series,
        detector,
        title="Interactive SVD Anomaly Detection - Demo"
    )

    # Save to HTML
    output_file = 'examples/plotly_single_scenario.html'
    fig.write_html(output_file)
    print(f"✓ Saved interactive plot to: {output_file}")
    print(f"  Open this file in a web browser to interact with the plot")

    return fig


def demo_multiple_scenarios():
    """Demo: Multiple scenarios comparison."""
    print("\nDemo 2: Multiple Scenarios Comparison")
    print("=" * 60)

    np.random.seed(42)

    scenarios_data = []

    # Scenario 1: Spike anomaly
    t = np.linspace(0, 10*np.pi, 500)
    ts1 = np.sin(t) + 0.1*np.random.randn(500)
    ts1[250] = 4.0
    det1 = SVDAnomalyDetector(window_size=15, threshold=2.5)
    det1.fit(ts1[:300])
    scenarios_data.append({
        'name': 'Spike Anomaly',
        'time_series': ts1,
        'anomalies': det1.predict(ts1),
        'scores': det1.get_anomaly_scores(ts1),
        'threshold': det1.mean + det1.threshold * det1.std
    })

    # Scenario 2: Gradual change
    ts2 = np.sin(t) + 0.1*np.random.randn(500)
    ts2[300:400] += 2.0
    det2 = SVDAnomalyDetector(window_size=15, threshold=2.5)
    det2.fit(ts2[:300])
    scenarios_data.append({
        'name': 'Gradual Change',
        'time_series': ts2,
        'anomalies': det2.predict(ts2),
        'scores': det2.get_anomaly_scores(ts2),
        'threshold': det2.mean + det2.threshold * det2.std
    })

    # Scenario 3: Noise burst
    ts3 = np.sin(t) + 0.1*np.random.randn(500)
    ts3[300:350] += 1.5*np.random.randn(50)
    det3 = SVDAnomalyDetector(window_size=15, threshold=2.5)
    det3.fit(ts3[:300])
    scenarios_data.append({
        'name': 'Noise Burst',
        'time_series': ts3,
        'anomalies': det3.predict(ts3),
        'scores': det3.get_anomaly_scores(ts3),
        'threshold': det3.mean + det3.threshold * det3.std
    })

    # Create visualization
    fig = create_multiple_scenarios_plot(scenarios_data)

    # Save to HTML
    output_file = 'examples/plotly_multiple_scenarios.html'
    fig.write_html(output_file)
    print(f"✓ Saved interactive comparison to: {output_file}")

    return fig


def demo_3d_landscape():
    """Demo: 3D anomaly landscape."""
    print("\nDemo 3: 3D Anomaly Score Landscape")
    print("=" * 60)

    np.random.seed(42)
    t = np.linspace(0, 10*np.pi, 500)
    time_series = np.sin(t) + 0.5*np.sin(3*t) + 0.1*np.random.randn(500)

    # Inject anomaly
    time_series[250] = 5.0

    # Base detector
    detector = SVDAnomalyDetector(window_size=20, threshold=2.5)
    detector.fit(time_series[:300])

    # Create 3D visualization
    print("  Computing scores for different window sizes (this may take a moment)...")
    fig = create_3d_anomaly_landscape(
        time_series,
        detector,
        window_range=(5, 40, 5)
    )

    # Save to HTML
    output_file = 'examples/plotly_3d_landscape.html'
    fig.write_html(output_file)
    print(f"✓ Saved 3D landscape to: {output_file}")

    return fig


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("INTERACTIVE PLOTLY VISUALIZATIONS FOR SVD ANOMALY DETECTION")
    print("=" * 60 + "\n")

    # Demo 1: Single scenario
    demo_single_scenario()

    # Demo 2: Multiple scenarios
    demo_multiple_scenarios()

    # Demo 3: 3D landscape
    demo_3d_landscape()

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. examples/plotly_single_scenario.html")
    print("  2. examples/plotly_multiple_scenarios.html")
    print("  3. examples/plotly_3d_landscape.html")
    print("\nOpen these HTML files in your web browser to:")
    print("  • Zoom and pan the plots")
    print("  • Hover over data points for details")
    print("  • Toggle traces on/off by clicking legend")
    print("  • Rotate and explore the 3D landscape")
    print("\nEnjoy exploring your anomaly detection results! 📊✨")


if __name__ == "__main__":
    main()

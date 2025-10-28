"""
Basic example of SVD-based anomaly detection for time series data.

This example demonstrates:
1. Generating synthetic time series data with anomalies
2. Training the SVD anomaly detector on normal data
3. Detecting anomalies in test data
4. Visualizing the results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import svd_anomaly_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svd_anomaly_detector import SVDAnomalyDetector


def generate_synthetic_data(n_points=1000, anomaly_ratio=0.05):
    """
    Generate synthetic time series data with anomalies.

    Creates a combination of sinusoidal patterns with noise,
    and injects random anomalies.
    """
    t = np.linspace(0, 10 * np.pi, n_points)

    # Normal pattern: combination of sine waves
    normal_signal = (
        np.sin(t) +
        0.5 * np.sin(3 * t) +
        0.3 * np.sin(5 * t) +
        0.1 * np.random.randn(n_points)  # Add noise
    )

    # Inject anomalies
    n_anomalies = int(n_points * anomaly_ratio)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    anomaly_indices.sort()

    time_series = normal_signal.copy()
    true_anomalies = np.zeros(n_points, dtype=int)

    for idx in anomaly_indices:
        # Create different types of anomalies
        anomaly_type = np.random.choice(['spike', 'drop', 'noise'])

        if anomaly_type == 'spike':
            time_series[idx] += np.random.uniform(3, 5)
        elif anomaly_type == 'drop':
            time_series[idx] -= np.random.uniform(3, 5)
        else:  # noise
            time_series[idx:idx+5] += np.random.randn(5) * 2

        true_anomalies[idx] = 1

    return time_series, true_anomalies


def main():
    print("SVD-based Time Series Anomaly Detection Example")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    print("\n1. Generating synthetic time series data...")
    time_series, true_anomalies = generate_synthetic_data(n_points=1000, anomaly_ratio=0.05)
    print(f"   Total points: {len(time_series)}")
    print(f"   True anomalies: {np.sum(true_anomalies)}")

    # Split into training (normal) and test data
    split_point = 700
    train_data = time_series[:split_point]
    test_data = time_series

    # Create and train detector
    print("\n2. Training SVD anomaly detector...")
    detector = SVDAnomalyDetector(
        window_size=20,
        rank=None,  # Auto-determine
        threshold=2.5
    )
    detector.fit(train_data)

    print(f"   Window size: {detector.window_size}")
    print(f"   Optimal rank: {detector.get_optimal_rank()}")
    print(f"   Explained variance (top 5): {detector.get_explained_variance_ratio()[:5]}")

    # Detect anomalies
    print("\n3. Detecting anomalies...")
    predicted_anomalies = detector.predict(test_data)
    anomaly_scores = detector.get_anomaly_scores(test_data)

    print(f"   Detected anomalies: {np.sum(predicted_anomalies)}")

    # Calculate metrics
    true_positives = np.sum((predicted_anomalies == 1) & (true_anomalies == 1))
    false_positives = np.sum((predicted_anomalies == 1) & (true_anomalies == 0))
    false_negatives = np.sum((predicted_anomalies == 0) & (true_anomalies == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n4. Results:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")

    # Visualization
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Time series with true anomalies
    axes[0].plot(time_series, label='Time Series', alpha=0.7)
    anomaly_points = np.where(true_anomalies == 1)[0]
    axes[0].scatter(anomaly_points, time_series[anomaly_points],
                   color='red', s=50, label='True Anomalies', zorder=5)
    axes[0].set_title('Original Time Series with True Anomalies')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Time series with detected anomalies
    axes[1].plot(time_series, label='Time Series', alpha=0.7)
    detected_points = np.where(predicted_anomalies == 1)[0]
    axes[1].scatter(detected_points, time_series[detected_points],
                   color='orange', s=50, label='Detected Anomalies', zorder=5)
    axes[1].axvline(x=split_point, color='green', linestyle='--',
                   label='Train/Test Split', alpha=0.5)
    axes[1].set_title('Time Series with Detected Anomalies (SVD-based)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Anomaly scores
    window_indices = np.arange(len(anomaly_scores))
    axes[2].plot(window_indices, anomaly_scores, label='Reconstruction Error', color='blue')
    threshold_line = detector.mean + detector.threshold * detector.std
    axes[2].axhline(y=threshold_line, color='red', linestyle='--',
                   label=f'Threshold ({threshold_line:.2f})', alpha=0.7)
    axes[2].set_title('Anomaly Scores (Reconstruction Error per Window)')
    axes[2].set_xlabel('Window Index')
    axes[2].set_ylabel('Reconstruction Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/anomaly_detection_results.png', dpi=150)
    print("   Saved visualization to 'examples/anomaly_detection_results.png'")

    # Show plot (comment out if running in non-interactive mode)
    # plt.show()

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()

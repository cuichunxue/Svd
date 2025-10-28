"""
Advanced example: Multi-variate time series anomaly detection.

This example shows how to use SVD for detecting anomalies in
multi-dimensional time series data, such as system metrics or
sensor readings from multiple sources.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svd_anomaly_detector import SVDAnomalyDetector


def generate_multivariate_data(n_points=1000, n_features=3):
    """
    Generate synthetic multi-variate time series data.

    Simulates system metrics like CPU, memory, and network usage.
    """
    t = np.linspace(0, 20 * np.pi, n_points)

    # Feature 1: CPU usage (periodic pattern)
    cpu = 50 + 20 * np.sin(t) + 10 * np.sin(3 * t) + 5 * np.random.randn(n_points)
    cpu = np.clip(cpu, 0, 100)

    # Feature 2: Memory usage (slowly increasing with noise)
    memory = 30 + 0.02 * t + 15 * np.sin(0.5 * t) + 5 * np.random.randn(n_points)
    memory = np.clip(memory, 0, 100)

    # Feature 3: Network traffic (bursty pattern)
    network = 20 + 10 * np.abs(np.sin(2 * t)) + 8 * np.random.randn(n_points)
    network = np.clip(network, 0, 100)

    # Stack features
    data = np.column_stack([cpu, memory, network])

    # Inject anomalies (system failures)
    anomaly_regions = [
        (200, 220),  # CPU spike
        (450, 470),  # Memory leak
        (750, 780),  # Network issue
    ]

    true_anomalies = np.zeros(n_points, dtype=int)

    for start, end in anomaly_regions:
        # CPU spike
        if start == 200:
            data[start:end, 0] += 40
        # Memory leak
        elif start == 450:
            data[start:end, 1] += np.linspace(0, 50, end - start)
        # Network issue
        elif start == 750:
            data[start:end, 2] += 50 * np.random.rand(end - start)

        true_anomalies[start:end] = 1

    return data, true_anomalies, ['CPU', 'Memory', 'Network']


def main():
    print("Advanced SVD Anomaly Detection: Multi-variate Time Series")
    print("=" * 70)

    np.random.seed(42)

    # Generate multi-variate data
    print("\n1. Generating multi-variate time series data...")
    data, true_anomalies, feature_names = generate_multivariate_data(n_points=1000, n_features=3)
    print(f"   Shape: {data.shape}")
    print(f"   Features: {feature_names}")
    print(f"   Anomalous regions: {np.sum(true_anomalies)} points")

    # Process each feature separately
    print("\n2. Detecting anomalies for each feature...")

    fig, axes = plt.subplots(len(feature_names) + 1, 1, figsize=(14, 12))

    all_anomalies = np.zeros(len(data), dtype=int)

    for i, feature_name in enumerate(feature_names):
        feature_data = data[:, i]

        # Train detector on first 40% of data (assumed normal)
        train_size = int(0.4 * len(feature_data))
        train_data = feature_data[:train_size]

        detector = SVDAnomalyDetector(
            window_size=15,
            rank=None,
            threshold=2.0
        )
        detector.fit(train_data)

        # Detect anomalies
        anomalies = detector.predict(feature_data)
        all_anomalies = np.logical_or(all_anomalies, anomalies)

        print(f"   {feature_name}: {np.sum(anomalies)} anomalies detected "
              f"(rank={detector.get_optimal_rank()})")

        # Plot
        axes[i].plot(feature_data, label=feature_name, alpha=0.7)
        detected_points = np.where(anomalies == 1)[0]
        axes[i].scatter(detected_points, feature_data[detected_points],
                       color='red', s=30, alpha=0.5, label='Detected Anomalies')
        axes[i].axvline(x=train_size, color='green', linestyle='--',
                       label='Train/Test Split', alpha=0.5)
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'{feature_name} Usage')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    # Combined view
    axes[-1].plot(true_anomalies, label='True Anomalies', alpha=0.7, linewidth=2)
    axes[-1].plot(all_anomalies, label='Detected Anomalies (Any Feature)',
                 alpha=0.7, linewidth=2)
    axes[-1].set_xlabel('Time')
    axes[-1].set_ylabel('Anomaly Flag')
    axes[-1].set_title('Combined Anomaly Detection Results')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/multivariate_anomaly_detection.png', dpi=150)
    print("\n   Saved visualization to 'examples/multivariate_anomaly_detection.png'")

    # Calculate overall metrics
    true_positives = np.sum((all_anomalies == 1) & (true_anomalies == 1))
    false_positives = np.sum((all_anomalies == 1) & (true_anomalies == 0))
    false_negatives = np.sum((all_anomalies == 0) & (true_anomalies == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n3. Overall Results:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")

    print("\n" + "=" * 70)
    print("Advanced example completed successfully!")


if __name__ == "__main__":
    main()

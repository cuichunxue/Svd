"""
Comprehensive Evaluation of SVD Anomaly Detection System

This script evaluates the detector's performance across multiple scenarios:
- Different time series patterns (seasonal, trend, periodic, noisy)
- Different anomaly types (spike, drop, gradual change, noise burst)
- Various difficulty levels (easy, medium, hard)
- Real-world-like scenarios

Author: Independent Evaluator
Date: 2025-10-29
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, Dict, List
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from svd_anomaly_detector import SVDAnomalyDetector


class TimeSeriesGenerator:
    """Generate various types of time series data for testing."""

    @staticmethod
    def seasonal(n_points=1000, noise_level=0.1):
        """Generate seasonal time series with daily and weekly patterns."""
        t = np.linspace(0, 10, n_points)
        # Daily pattern (fast oscillation)
        daily = 2 * np.sin(2 * np.pi * t)
        # Weekly pattern (slow oscillation)
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        # Noise
        noise = noise_level * np.random.randn(n_points)
        return daily + weekly + noise

    @staticmethod
    def trend(n_points=1000, noise_level=0.1):
        """Generate time series with linear trend."""
        t = np.linspace(0, 10, n_points)
        trend = 0.5 * t
        seasonal = np.sin(2 * np.pi * t)
        noise = noise_level * np.random.randn(n_points)
        return trend + seasonal + noise

    @staticmethod
    def periodic(n_points=1000, noise_level=0.1):
        """Generate simple periodic time series."""
        t = np.linspace(0, 20 * np.pi, n_points)
        signal = np.sin(t) + 0.5 * np.sin(3 * t)
        noise = noise_level * np.random.randn(n_points)
        return signal + noise

    @staticmethod
    def random_walk(n_points=1000, noise_level=0.1):
        """Generate random walk time series."""
        steps = np.random.randn(n_points) * noise_level
        return np.cumsum(steps)

    @staticmethod
    def complex_pattern(n_points=1000, noise_level=0.1):
        """Generate complex time series with multiple components."""
        t = np.linspace(0, 10, n_points)
        # Multiple frequencies
        comp1 = 2 * np.sin(2 * np.pi * t)
        comp2 = 1.5 * np.sin(2 * np.pi * t * 3)
        comp3 = np.sin(2 * np.pi * t * 7)
        # Trend
        trend = 0.2 * t
        # Noise
        noise = noise_level * np.random.randn(n_points)
        return comp1 + comp2 + comp3 + trend + noise


class AnomalyInjector:
    """Inject various types of anomalies into time series."""

    @staticmethod
    def spike(data, position, magnitude=5.0):
        """Inject a spike anomaly."""
        data_copy = data.copy()
        data_copy[position] += magnitude
        anomaly_mask = np.zeros(len(data), dtype=int)
        anomaly_mask[position] = 1
        return data_copy, anomaly_mask

    @staticmethod
    def drop(data, position, magnitude=5.0):
        """Inject a drop anomaly."""
        data_copy = data.copy()
        data_copy[position] -= magnitude
        anomaly_mask = np.zeros(len(data), dtype=int)
        anomaly_mask[position] = 1
        return data_copy, anomaly_mask

    @staticmethod
    def gradual_change(data, start, duration, magnitude=3.0):
        """Inject a gradual change (level shift)."""
        data_copy = data.copy()
        end = start + duration
        data_copy[start:end] += magnitude
        anomaly_mask = np.zeros(len(data), dtype=int)
        anomaly_mask[start:end] = 1
        return data_copy, anomaly_mask

    @staticmethod
    def noise_burst(data, start, duration, magnitude=2.0):
        """Inject a burst of noise."""
        data_copy = data.copy()
        end = start + duration
        data_copy[start:end] += magnitude * np.random.randn(duration)
        anomaly_mask = np.zeros(len(data), dtype=int)
        anomaly_mask[start:end] = 1
        return data_copy, anomaly_mask

    @staticmethod
    def pattern_change(data, start, duration):
        """Inject a pattern change (frequency shift)."""
        data_copy = data.copy()
        end = start + duration
        t = np.linspace(0, 2 * np.pi * duration / 10, duration)
        # Replace with different frequency pattern
        data_copy[start:end] = 3 * np.sin(10 * t)
        anomaly_mask = np.zeros(len(data), dtype=int)
        anomaly_mask[start:end] = 1
        return data_copy, anomaly_mask

    @staticmethod
    def multiple_anomalies(data, anomaly_configs):
        """Inject multiple anomalies of different types."""
        data_copy = data.copy()
        anomaly_mask = np.zeros(len(data), dtype=int)

        for config in anomaly_configs:
            anom_type = config['type']

            if anom_type == 'spike':
                data_copy, mask = AnomalyInjector.spike(
                    data_copy, config['position'], config.get('magnitude', 5.0)
                )
            elif anom_type == 'drop':
                data_copy, mask = AnomalyInjector.drop(
                    data_copy, config['position'], config.get('magnitude', 5.0)
                )
            elif anom_type == 'gradual':
                data_copy, mask = AnomalyInjector.gradual_change(
                    data_copy, config['start'], config['duration'], config.get('magnitude', 3.0)
                )
            elif anom_type == 'noise_burst':
                data_copy, mask = AnomalyInjector.noise_burst(
                    data_copy, config['start'], config['duration'], config.get('magnitude', 2.0)
                )
            elif anom_type == 'pattern_change':
                data_copy, mask = AnomalyInjector.pattern_change(
                    data_copy, config['start'], config['duration']
                )

            anomaly_mask = np.logical_or(anomaly_mask, mask).astype(int)

        return data_copy, anomaly_mask


class EvaluationMetrics:
    """Calculate various evaluation metrics."""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate precision, recall, F1, and other metrics."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'fpr': fpr
        }


class SVDEvaluator:
    """Comprehensive evaluator for SVD anomaly detection."""

    def __init__(self, n_points=1000):
        self.n_points = n_points
        self.results = []

    def evaluate_scenario(self, name, data, true_anomalies, window_size=20,
                         rank=None, threshold=2.5, train_ratio=0.6):
        """Evaluate a single scenario."""
        print(f"\n  Testing: {name}")

        try:
            # Split data
            train_size = int(len(data) * train_ratio)
            train_data = data[:train_size]

            # Train detector
            detector = SVDAnomalyDetector(
                window_size=window_size,
                rank=rank,
                threshold=threshold
            )
            detector.fit(train_data)

            # Predict
            predicted = detector.predict(data)
            scores = detector.get_anomaly_scores(data)

            # Calculate metrics
            metrics = EvaluationMetrics.calculate_metrics(true_anomalies, predicted)

            # Additional info
            metrics['scenario'] = name
            metrics['window_size'] = window_size
            metrics['rank_used'] = detector.get_optimal_rank()
            metrics['threshold'] = threshold
            metrics['train_size'] = train_size
            metrics['anomalies_detected'] = int(np.sum(predicted))
            metrics['anomalies_true'] = int(np.sum(true_anomalies))

            print(f"    Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")

            return {
                'metrics': metrics,
                'data': data,
                'true_anomalies': true_anomalies,
                'predicted': predicted,
                'scores': scores,
                'detector': detector
            }

        except Exception as e:
            print(f"    ERROR: {e}")
            return None

    def run_comprehensive_evaluation(self):
        """Run evaluation on multiple scenarios."""
        print("=" * 80)
        print("COMPREHENSIVE EVALUATION OF SVD ANOMALY DETECTION")
        print("=" * 80)

        np.random.seed(42)

        # Test 1: Different time series patterns with spike anomalies
        print("\n[TEST 1] Different Time Series Patterns with Spike Anomalies")
        print("-" * 80)

        patterns = {
            'Seasonal': TimeSeriesGenerator.seasonal(self.n_points),
            'Trend': TimeSeriesGenerator.trend(self.n_points),
            'Periodic': TimeSeriesGenerator.periodic(self.n_points),
            'Random Walk': TimeSeriesGenerator.random_walk(self.n_points),
            'Complex Pattern': TimeSeriesGenerator.complex_pattern(self.n_points)
        }

        for pattern_name, base_data in patterns.items():
            # Inject spike anomaly
            anomaly_pos = self.n_points // 2
            data, mask = AnomalyInjector.spike(base_data, anomaly_pos, magnitude=5.0)

            result = self.evaluate_scenario(
                f"{pattern_name} + Spike",
                data, mask,
                window_size=20,
                threshold=2.5
            )
            if result:
                self.results.append(result)

        # Test 2: Different anomaly types on periodic data
        print("\n[TEST 2] Different Anomaly Types on Periodic Data")
        print("-" * 80)

        base_data = TimeSeriesGenerator.periodic(self.n_points)

        # Spike
        data, mask = AnomalyInjector.spike(base_data, 500, magnitude=5.0)
        result = self.evaluate_scenario("Periodic + Large Spike", data, mask)
        if result:
            self.results.append(result)

        # Drop
        data, mask = AnomalyInjector.drop(base_data, 500, magnitude=5.0)
        result = self.evaluate_scenario("Periodic + Large Drop", data, mask)
        if result:
            self.results.append(result)

        # Gradual change
        data, mask = AnomalyInjector.gradual_change(base_data, 400, 100, magnitude=3.0)
        result = self.evaluate_scenario("Periodic + Gradual Change", data, mask)
        if result:
            self.results.append(result)

        # Noise burst
        data, mask = AnomalyInjector.noise_burst(base_data, 400, 50, magnitude=2.0)
        result = self.evaluate_scenario("Periodic + Noise Burst", data, mask)
        if result:
            self.results.append(result)

        # Pattern change
        data, mask = AnomalyInjector.pattern_change(base_data, 400, 100)
        result = self.evaluate_scenario("Periodic + Pattern Change", data, mask)
        if result:
            self.results.append(result)

        # Test 3: Difficulty levels (different anomaly magnitudes)
        print("\n[TEST 3] Different Difficulty Levels (Anomaly Magnitudes)")
        print("-" * 80)

        base_data = TimeSeriesGenerator.seasonal(self.n_points, noise_level=0.2)

        for magnitude, difficulty in [(8.0, 'Easy'), (4.0, 'Medium'), (2.0, 'Hard')]:
            data, mask = AnomalyInjector.spike(base_data, 500, magnitude=magnitude)
            result = self.evaluate_scenario(
                f"Seasonal + Spike [{difficulty}]",
                data, mask,
                threshold=2.5
            )
            if result:
                self.results.append(result)

        # Test 4: Multiple anomalies
        print("\n[TEST 4] Multiple Anomalies")
        print("-" * 80)

        base_data = TimeSeriesGenerator.complex_pattern(self.n_points)

        anomaly_configs = [
            {'type': 'spike', 'position': 200, 'magnitude': 5.0},
            {'type': 'drop', 'position': 400, 'magnitude': 4.0},
            {'type': 'gradual', 'start': 600, 'duration': 50, 'magnitude': 3.0},
            {'type': 'noise_burst', 'start': 800, 'duration': 30, 'magnitude': 2.5}
        ]

        data, mask = AnomalyInjector.multiple_anomalies(base_data, anomaly_configs)
        result = self.evaluate_scenario("Multiple Anomaly Types", data, mask, threshold=2.0)
        if result:
            self.results.append(result)

        # Test 5: Parameter sensitivity
        print("\n[TEST 5] Parameter Sensitivity Analysis")
        print("-" * 80)

        base_data = TimeSeriesGenerator.periodic(self.n_points)
        data, mask = AnomalyInjector.spike(base_data, 500, magnitude=4.0)

        # Different window sizes
        for ws in [10, 20, 40]:
            result = self.evaluate_scenario(
                f"Parameter Test: window_size={ws}",
                data, mask,
                window_size=ws,
                threshold=2.5
            )
            if result:
                self.results.append(result)

        # Different thresholds
        for th in [1.5, 2.5, 3.5]:
            result = self.evaluate_scenario(
                f"Parameter Test: threshold={th}",
                data, mask,
                window_size=20,
                threshold=th
            )
            if result:
                self.results.append(result)

        # Test 6: Noisy data
        print("\n[TEST 6] Performance with Different Noise Levels")
        print("-" * 80)

        for noise_level in [0.05, 0.2, 0.5]:
            base_data = TimeSeriesGenerator.seasonal(self.n_points, noise_level=noise_level)
            data, mask = AnomalyInjector.spike(base_data, 500, magnitude=5.0)
            result = self.evaluate_scenario(
                f"Noise Level: {noise_level}",
                data, mask,
                threshold=2.5
            )
            if result:
                self.results.append(result)

        return self.results

    def generate_summary_report(self):
        """Generate summary statistics and report."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        if not self.results:
            print("No results to summarize.")
            return None

        # Extract metrics
        all_metrics = [r['metrics'] for r in self.results]

        # Overall statistics
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])

        print(f"\nOverall Performance (Average across {len(all_metrics)} scenarios):")
        print(f"  Precision: {avg_precision:.3f} (±{np.std([m['precision'] for m in all_metrics]):.3f})")
        print(f"  Recall:    {avg_recall:.3f} (±{np.std([m['recall'] for m in all_metrics]):.3f})")
        print(f"  F1 Score:  {avg_f1:.3f} (±{np.std([m['f1_score'] for m in all_metrics]):.3f})")
        print(f"  Accuracy:  {avg_accuracy:.3f} (±{np.std([m['accuracy'] for m in all_metrics]):.3f})")

        # Best and worst performers
        best_f1 = max(all_metrics, key=lambda x: x['f1_score'])
        worst_f1 = min(all_metrics, key=lambda x: x['f1_score'])

        print(f"\nBest Performance (F1={best_f1['f1_score']:.3f}):")
        print(f"  Scenario: {best_f1['scenario']}")
        print(f"  Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}")

        print(f"\nWorst Performance (F1={worst_f1['f1_score']:.3f}):")
        print(f"  Scenario: {worst_f1['scenario']}")
        print(f"  Precision: {worst_f1['precision']:.3f}, Recall: {worst_f1['recall']:.3f}")

        # Category analysis
        print("\n" + "-" * 80)
        print("Performance by Category:")
        print("-" * 80)

        categories = {}
        for m in all_metrics:
            # Extract category from scenario name
            if 'Spike' in m['scenario'] or 'Drop' in m['scenario']:
                cat = 'Point Anomalies'
            elif 'Gradual' in m['scenario'] or 'Pattern' in m['scenario']:
                cat = 'Contextual Anomalies'
            elif 'Noise' in m['scenario']:
                cat = 'Noise-based Anomalies'
            elif 'Parameter' in m['scenario']:
                cat = 'Parameter Sensitivity'
            elif 'Multiple' in m['scenario']:
                cat = 'Multiple Anomalies'
            else:
                cat = 'Time Series Patterns'

            if cat not in categories:
                categories[cat] = []
            categories[cat].append(m)

        for cat, metrics in categories.items():
            cat_f1 = np.mean([m['f1_score'] for m in metrics])
            cat_prec = np.mean([m['precision'] for m in metrics])
            cat_rec = np.mean([m['recall'] for m in metrics])
            print(f"\n{cat}:")
            print(f"  F1: {cat_f1:.3f}, Precision: {cat_prec:.3f}, Recall: {cat_rec:.3f}")
            print(f"  (n={len(metrics)} scenarios)")

        summary = {
            'overall': {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'avg_accuracy': avg_accuracy,
                'n_scenarios': len(all_metrics)
            },
            'best': best_f1,
            'worst': worst_f1,
            'categories': {cat: {
                'f1': float(np.mean([m['f1_score'] for m in metrics])),
                'precision': float(np.mean([m['precision'] for m in metrics])),
                'recall': float(np.mean([m['recall'] for m in metrics])),
                'n_scenarios': len(metrics)
            } for cat, metrics in categories.items()},
            'all_metrics': all_metrics
        }

        return summary

    def visualize_results(self, output_dir='evaluation'):
        """Generate comprehensive visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        # Plot 1: Performance comparison across scenarios
        metrics = [r['metrics'] for r in self.results]
        scenarios = [m['scenario'] for m in metrics]
        f1_scores = [m['f1_score'] for m in metrics]
        precisions = [m['precision'] for m in metrics]
        recalls = [m['recall'] for m in metrics]

        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(scenarios))
        width = 0.25

        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)

        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Comparison Across All Scenarios', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_comparison.png', dpi=150)
        print(f"\nSaved: {output_dir}/performance_comparison.png")

        # Plot 2: Example visualizations for selected scenarios
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.flatten()

        # Select 6 interesting scenarios
        selected_indices = [0, 5, 9, 14, 18, 21] if len(self.results) > 21 else list(range(min(6, len(self.results))))

        for idx, result_idx in enumerate(selected_indices):
            if result_idx >= len(self.results):
                break

            result = self.results[result_idx]
            ax = axes[idx]

            data = result['data']
            true_anom = result['true_anomalies']
            pred_anom = result['predicted']

            # Plot time series
            ax.plot(data, label='Time Series', alpha=0.6, linewidth=1)

            # Plot true anomalies
            true_points = np.where(true_anom == 1)[0]
            if len(true_points) > 0:
                ax.scatter(true_points, data[true_points],
                          color='red', s=30, label='True Anomaly',
                          marker='o', zorder=5, alpha=0.7)

            # Plot detected anomalies
            det_points = np.where(pred_anom == 1)[0]
            if len(det_points) > 0:
                ax.scatter(det_points, data[det_points],
                          color='orange', s=20, label='Detected',
                          marker='x', zorder=4, alpha=0.7)

            result_metrics = result['metrics']
            title = f"{result_metrics['scenario']}\nF1={result_metrics['f1_score']:.2f}, P={result_metrics['precision']:.2f}, R={result_metrics['recall']:.2f}"
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/example_scenarios.png', dpi=150)
        print(f"Saved: {output_dir}/example_scenarios.png")

        # Plot 3: Distribution of metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist([m['f1_score'] for m in metrics], bins=15, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('F1 Score Distribution')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean([m['f1_score'] for m in metrics]), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()

        axes[0, 1].hist([m['precision'] for m in metrics], bins=15, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Precision Distribution')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean([m['precision'] for m in metrics]), color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()

        axes[1, 0].hist([m['recall'] for m in metrics], bins=15, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Recall Distribution')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean([m['recall'] for m in metrics]), color='red', linestyle='--', label='Mean')
        axes[1, 0].legend()

        # Scatter: Precision vs Recall
        axes[1, 1].scatter([m['precision'] for m in metrics], [m['recall'] for m in metrics],
                          alpha=0.6, s=50)
        axes[1, 1].set_xlabel('Precision')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Perfect')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_distribution.png', dpi=150)
        print(f"Saved: {output_dir}/metrics_distribution.png")

        plt.close('all')


def main():
    """Run comprehensive evaluation."""
    print("Starting comprehensive evaluation...")
    print("This will test the SVD anomaly detector across multiple scenarios.\n")

    # Create evaluator
    evaluator = SVDEvaluator(n_points=1000)

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()

    # Generate summary
    summary = evaluator.generate_summary_report()

    # Visualize
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    evaluator.visualize_results()

    # Save summary to JSON
    if summary:
        summary_file = 'evaluation/evaluation_summary.json'

        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=convert_to_python_types)
        print(f"\nSaved summary to: {summary_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - evaluation/performance_comparison.png")
    print("  - evaluation/example_scenarios.png")
    print("  - evaluation/metrics_distribution.png")
    print("  - evaluation/evaluation_summary.json")
    print("\nConclusion:")
    if summary:
        avg_f1 = summary['overall']['avg_f1']
        if avg_f1 > 0.7:
            print(f"  ✓ GOOD: Average F1 score of {avg_f1:.3f} indicates strong performance")
        elif avg_f1 > 0.5:
            print(f"  ~ MODERATE: Average F1 score of {avg_f1:.3f} shows acceptable performance")
        else:
            print(f"  ✗ POOR: Average F1 score of {avg_f1:.3f} suggests room for improvement")


if __name__ == "__main__":
    main()

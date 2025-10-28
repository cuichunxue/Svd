"""
Simple test script to verify SVD anomaly detector installation and basic functionality.
"""

import sys
import numpy as np

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from src.svd_anomaly_detector import SVDAnomalyDetector, detect_anomalies_svd
        print("  OK: Successfully imported SVDAnomalyDetector and detect_anomalies_svd")
        return True
    except ImportError as e:
        print(f"  FAILED: Could not import modules: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of the anomaly detector."""
    print("\nTesting basic functionality...")
    try:
        from src.svd_anomaly_detector import SVDAnomalyDetector

        # Generate simple test data
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, 200)
        data = np.sin(t) + 0.1 * np.random.randn(200)

        # Add a clear anomaly
        data[100] = 5.0

        # Create and train detector
        detector = SVDAnomalyDetector(window_size=10, threshold=2.5)
        detector.fit(data[:150])

        # Predict anomalies
        anomalies = detector.predict(data)
        scores = detector.get_anomaly_scores(data)

        # Verify results
        assert len(anomalies) == len(data), "Anomaly array length mismatch"
        assert len(scores) == len(data) - detector.window_size + 1, "Scores array length mismatch"
        assert np.sum(anomalies) > 0, "No anomalies detected in data with known anomaly"

        print(f"  OK: Detector created and trained successfully")
        print(f"  OK: Detected {np.sum(anomalies)} anomalous points")
        print(f"  OK: Optimal rank: {detector.get_optimal_rank()}")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test the convenience function."""
    print("\nTesting convenience function...")
    try:
        from src.svd_anomaly_detector import detect_anomalies_svd

        # Generate test data
        np.random.seed(42)
        data = np.random.randn(100)
        data[50] = 10.0  # Add anomaly

        # Use convenience function
        anomalies, scores = detect_anomalies_svd(data, window_size=5)

        assert len(anomalies) == len(data), "Anomaly array length mismatch"
        print(f"  OK: Convenience function works correctly")
        print(f"  OK: Detected {np.sum(anomalies)} anomalies")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SVD Anomaly Detector - Installation Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Basic Functionality Test", test_basic_functionality()))
    results.append(("Convenience Function Test", test_convenience_function()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\nAll tests passed! Installation is successful.")
        print("\nNext steps:")
        print("  1. Run 'python examples/basic_example.py' for a complete example")
        print("  2. Read QUICKSTART.md for quick usage guide")
        print("  3. Read README.md for detailed documentation")
        return 0
    else:
        print("\nSome tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

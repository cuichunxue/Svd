"""
SVD-based Time Series Anomaly Detection Package
"""

from .svd_anomaly_detector import SVDAnomalyDetector, detect_anomalies_svd

__version__ = "1.0.0"
__all__ = ["SVDAnomalyDetector", "detect_anomalies_svd"]

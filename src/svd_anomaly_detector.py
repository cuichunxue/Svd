"""
SVD-based Time Series Anomaly Detection

This module implements anomaly detection for time series data using
Singular Value Decomposition (SVD). The method creates a trajectory matrix
from the time series using a sliding window approach, applies SVD to extract
the principal components, and detects anomalies based on reconstruction error.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import svd


class SVDAnomalyDetector:
    """
    Time series anomaly detector using Singular Value Decomposition.

    The detector uses a sliding window to create a trajectory matrix,
    applies SVD for dimensionality reduction, and identifies anomalies
    based on reconstruction error.

    Parameters
    ----------
    window_size : int
        Size of the sliding window for creating trajectory matrix
    rank : int, optional
        Number of singular values to retain (if None, automatically determined)
    threshold : float
        Threshold multiplier for anomaly detection (default: 3.0)
    """

    def __init__(self, window_size: int, rank: Optional[int] = None, threshold: float = 3.0):
        self.window_size = window_size
        self.rank = rank
        self.threshold = threshold
        self.U = None
        self.s = None
        self.Vt = None
        self.mean = None
        self.std = None

    def _create_trajectory_matrix(self, time_series: np.ndarray) -> np.ndarray:
        """
        Create trajectory matrix using sliding window (Hankel matrix).

        Parameters
        ----------
        time_series : np.ndarray
            Input time series data (1D array)

        Returns
        -------
        np.ndarray
            Trajectory matrix of shape (window_size, n_windows)
        """
        n = len(time_series)
        m = n - self.window_size + 1

        if m <= 0:
            raise ValueError(f"Time series length ({n}) must be greater than window_size ({self.window_size})")

        trajectory_matrix = np.zeros((self.window_size, m))
        for i in range(m):
            trajectory_matrix[:, i] = time_series[i:i + self.window_size]

        return trajectory_matrix

    def fit(self, time_series: np.ndarray) -> 'SVDAnomalyDetector':
        """
        Fit the SVD model to normal time series data.

        Parameters
        ----------
        time_series : np.ndarray
            Training time series data (assumed to be normal/non-anomalous)

        Returns
        -------
        self
        """
        # Create trajectory matrix
        trajectory_matrix = self._create_trajectory_matrix(time_series)

        # Apply SVD
        self.U, self.s, self.Vt = svd(trajectory_matrix, full_matrices=False)

        # Determine rank if not specified
        if self.rank is None:
            # Use elbow method: retain components that explain 95% of variance
            cumulative_variance = np.cumsum(self.s ** 2) / np.sum(self.s ** 2)
            self.rank = np.argmax(cumulative_variance >= 0.95) + 1
            self.rank = max(1, min(self.rank, len(self.s)))

        # Calculate reconstruction errors for training data to set threshold
        reconstruction_errors = self._calculate_reconstruction_errors(time_series)
        self.mean = np.mean(reconstruction_errors)
        self.std = np.std(reconstruction_errors)

        return self

    def _calculate_reconstruction_errors(self, time_series: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each window in the time series.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series data

        Returns
        -------
        np.ndarray
            Reconstruction errors for each window
        """
        trajectory_matrix = self._create_trajectory_matrix(time_series)

        # Reconstruct using top-k singular values
        U_k = self.U[:, :self.rank]
        s_k = self.s[:self.rank]
        Vt_k = self.Vt[:self.rank, :]

        reconstructed = U_k @ np.diag(s_k) @ Vt_k

        # Calculate reconstruction error for each window
        errors = np.linalg.norm(trajectory_matrix - reconstructed, axis=0)

        return errors

    def predict(self, time_series: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in time series data.

        Parameters
        ----------
        time_series : np.ndarray
            Time series data to check for anomalies

        Returns
        -------
        np.ndarray
            Binary array (1 for anomaly, 0 for normal) for each time point
        """
        if self.U is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        reconstruction_errors = self._calculate_reconstruction_errors(time_series)

        # Detect anomalies using threshold based on training statistics
        anomaly_threshold = self.mean + self.threshold * self.std

        # Create anomaly array for full time series
        n = len(time_series)
        anomalies = np.zeros(n, dtype=int)

        # Mark anomalies (windows with high reconstruction error)
        for i, error in enumerate(reconstruction_errors):
            if error > anomaly_threshold:
                # Mark all points in this window as anomalous
                start_idx = i
                end_idx = min(i + self.window_size, n)
                anomalies[start_idx:end_idx] = 1

        return anomalies

    def get_anomaly_scores(self, time_series: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (reconstruction errors) for time series.

        Parameters
        ----------
        time_series : np.ndarray
            Time series data

        Returns
        -------
        np.ndarray
            Anomaly scores for each window
        """
        if self.U is None:
            raise ValueError("Model must be fitted before getting scores. Call fit() first.")

        return self._calculate_reconstruction_errors(time_series)

    def get_optimal_rank(self) -> int:
        """Get the rank (number of components) used for reconstruction."""
        return self.rank

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each singular value."""
        if self.s is None:
            raise ValueError("Model must be fitted first.")

        return (self.s ** 2) / np.sum(self.s ** 2)


def detect_anomalies_svd(
    time_series: np.ndarray,
    window_size: int = 10,
    rank: Optional[int] = None,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to detect anomalies in time series using SVD.

    Parameters
    ----------
    time_series : np.ndarray
        Input time series data
    window_size : int
        Size of the sliding window
    rank : int, optional
        Number of singular values to retain
    threshold : float
        Threshold multiplier for anomaly detection

    Returns
    -------
    tuple of (anomalies, scores)
        anomalies : Binary array indicating anomalies
        scores : Anomaly scores for each window
    """
    detector = SVDAnomalyDetector(window_size=window_size, rank=rank, threshold=threshold)
    detector.fit(time_series)
    anomalies = detector.predict(time_series)
    scores = detector.get_anomaly_scores(time_series)

    return anomalies, scores

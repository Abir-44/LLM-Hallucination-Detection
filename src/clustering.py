"""
Clustering and Discretization Module
Part of Member 3's pipeline: K-Means Discretization of LLM Internal Activations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class KMeansDiscretizer:
    """
    K-Means clustering wrapper to discretize continuous activation vectors
    into discrete activation state IDs (0 to K-1).
    """

    def __init__(
        self,
        n_clusters: int = 20,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "KMeansDiscretizer":
        """
        Fits the K-Means model on activation vectors.
        """
        self.model.fit(X)
        self.is_fitted = True
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits K-Means and returns discrete cluster IDs for each sample.
        """
        labels = self.model.fit_predict(X)
        self.is_fitted = True
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts discrete cluster ID for continuous activation vectors.
        """
        if not self.is_fitted:
            raise ValueError("KMeansDiscretizer must be fitted before calling predict().")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes key clustering evaluation metrics:
        - Inertia (Sum of squared distances to closest centroid)
        - Silhouette Score
        - Davies-Bouldin Index (lower is better)
        - Calinski-Harabasz Index (higher is better)
        - Cluster size distribution
        """
        if labels is None:
            labels = self.predict(X)

        inertia = float(self.model.inertia_)
        sil_score = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
        db_score = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
        ch_score = float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0

        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = {int(k): int(v) for k, v in zip(unique, counts)}

        metrics = {
            "n_clusters": self.n_clusters,
            "inertia": inertia,
            "silhouette_score": sil_score,
            "davies_bouldin_index": db_score,
            "calinski_harabasz_index": ch_score,
            "min_cluster_size": int(min(counts)),
            "max_cluster_size": int(max(counts)),
            "mean_cluster_size": float(np.mean(counts)),
            "cluster_distribution": cluster_counts
        }
        return metrics

    def save(self, filepath: str = "models/kmeans_k20.joblib") -> str:
        """
        Serializes trained KMeans model using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Saved KMeans model to '{filepath}'.")
        return filepath

    @classmethod
    def load(cls, filepath: str = "models/kmeans_k20.joblib") -> "KMeansDiscretizer":
        """
        Loads pre-trained KMeans model from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        kmeans_raw = joblib.load(filepath)
        discretizer = cls(n_clusters=kmeans_raw.n_clusters, random_state=kmeans_raw.random_state)
        discretizer.model = kmeans_raw
        discretizer.is_fitted = True
        return discretizer

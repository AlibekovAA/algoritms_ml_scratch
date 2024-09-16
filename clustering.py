from typing import Optional, Callable, Union
import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 10, n_init: int = 3, random_state: int = 42):
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.n_init: int = n_init
        self.random_state: int = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key,
                           value in self.__dict__.items())
        return f"MyKMeans class: {params}"

    def _initialize_centroids(self, X: pd.DataFrame) -> np.ndarray:
        """Инициализация случайных центроидов для каждого кластера."""
        np.random.seed(self.random_state)
        min_vals = X.min(axis=0).values
        max_vals = X.max(axis=0).values
        centroids = np.random.uniform(
            min_vals, max_vals, (self.n_clusters, X.shape[1]))
        return centroids

    def _assign_clusters(self, X: pd.DataFrame, centroids: np.ndarray) -> np.ndarray:
        """Назначение кластеров для каждой точки данных."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        return clusters

    def _compute_centroids(self, X: pd.DataFrame, labels: np.ndarray, old_centroids: np.ndarray) -> np.ndarray:
        """Вычисление новых центроидов как среднее значение точек в каждом кластере."""
        centroids = []
        for k in range(self.n_clusters):
            cluster_points = X.values[labels == k]
            if len(cluster_points) == 0:
                centroids.append(old_centroids[k])
            else:
                new_centroid = cluster_points.mean(axis=0)
                centroids.append(new_centroid)
        return np.array(centroids)

    def _compute_wcss(self, X: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Вычисление суммы квадратов внутрикластерных расстояний до центроидов (WCSS)."""
        wcss = 0.0
        X = X.values if isinstance(X, pd.DataFrame) else X
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] > 0:
                wcss += np.sum((cluster_points - centroids[k]) ** 2)
        return wcss

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Обучение модели K-means."""
        X = pd.DataFrame(X)
        best_wcss = np.inf
        best_centroids = None

        for i in range(self.n_init):
            centroids = self._initialize_centroids(X)
            for j in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._compute_centroids(X, labels, centroids)

                if np.all(centroids == new_centroids):
                    break

                centroids = new_centroids

            wcss = self._compute_wcss(X, labels, centroids)

            if wcss < best_wcss:
                best_wcss = wcss
                best_centroids = centroids

        self.cluster_centers_ = best_centroids
        self.inertia_ = best_wcss

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Предсказание кластеров для новых точек данных на основе обученной модели."""
        X = pd.DataFrame(X)
        labels = self._assign_clusters(X, self.cluster_centers_)
        return labels

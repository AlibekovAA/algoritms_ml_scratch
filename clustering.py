from typing import Optional, Callable, Union, List
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


class MyDBSCAN:
    def __init__(self, eps: float = 3, min_samples: int = 3, metric: str = 'euclidean') -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.distance_function = self._select_metric(metric)

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f"MyDBSCAN class: {params}"

    def _select_metric(self, metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
        if metric == 'euclidean':
            return self._euclidean_distance
        elif metric == 'chebyshev':
            return self._chebyshev_distance
        elif metric == 'manhattan':
            return self._manhattan_distance
        elif metric == 'cosine':
            return self._cosine_distance
        else:
            raise ValueError("Unsupported metric. Choose from: 'euclidean', 'chebyshev', 'manhattan', 'cosine'.")

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _chebyshev_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.max(np.abs(point1 - point2))

    def _manhattan_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sum(np.abs(point1 - point2))

    def _cosine_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        dot_product = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        return 1 - (dot_product / (norm1 * norm2))

    def region_query(self, X: np.ndarray, point_idx: int) -> List[int]:
        neighbors = []
        for idx in range(len(X)):
            if idx != point_idx and self.distance_function(X[point_idx], X[idx]) <= self.eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(self, X: np.ndarray, labels: List[int], point_idx: int, neighbors: List[int], cluster_id: int) -> None:
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self.region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples - 1:
                    neighbors.extend(new_neighbors)
            i += 1

    def fit_predict(self, X: pd.DataFrame) -> List[int]:
        X = X.to_numpy()
        labels = [0] * len(X)
        cluster_id = 0

        for point_idx in range(len(X)):
            if labels[point_idx] != 0:
                continue

            neighbors = self.region_query(X, point_idx)
            if len(neighbors) < self.min_samples - 1:
                labels[point_idx] = -1
            else:
                cluster_id += 1
                self.expand_cluster(X, labels, point_idx, neighbors, cluster_id)

        return labels

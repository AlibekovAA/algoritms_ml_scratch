import numpy as np
import pandas as pd
from typing import Union


class MyKNNClf:

    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform") -> None:
        self.k: int = k
        self.metric: str = metric
        self.weight: str = weight

        self.train_size: Union[tuple, int] = 0
        self.train_x: Union[np.ndarray, None] = None
        self.train_y: Union[np.ndarray, None] = None

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key,
                           value in self.__dict__.items())
        return f"MyKNNClf class: {params}"

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series) -> None:
        self.train_x = self._asarray(train_x)
        self.train_y = self._asarray(train_y)
        self.train_size = self.train_x.shape

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_hat = self.predict_proba(X)
        return (y_hat >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._asarray(X)
        distance = self._pairwise_distance(X, self.train_x)
        k_nearest_indices = np.argsort(distance, axis=1)[:, :self.k]
        neighbors = self.train_y[k_nearest_indices]

        if self.weight == "uniform":
            return np.mean(neighbors, axis=1)

        elif self.weight == "rank":
            ranks = np.arange(1, self.k + 1)[::-1]
            weights = np.tile(ranks, (X.shape[0], 1))
            weighted_sum = np.sum(neighbors * weights, axis=1)
            weight_sum = np.sum(weights, axis=1)
            return weighted_sum / weight_sum

        elif self.weight == "distance":
            distances = distance[np.argsort(distance, axis=1)[:, :self.k]]
            weights = 1 / (distances + 1e-5)
            weighted_sum = np.sum(neighbors * weights, axis=1)
            return weighted_sum / np.sum(weights, axis=1)

        else:
            raise ValueError(f"Invalid weight: {self.weight}")

    def _pairwise_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=2))

        elif self.metric == "chebyshev":
            return np.max(np.abs(x[:, np.newaxis] - y), axis=2)

        elif self.metric == "manhattan":
            return np.sum(np.abs(x[:, np.newaxis] - y), axis=2)

        elif self.metric == "cosine":
            x_norm = np.linalg.norm(x, axis=1)
            y_norm = np.linalg.norm(y, axis=1)
            similarity = np.dot(x, y.T) / (x_norm[:, np.newaxis] * y_norm)
            return 1 - similarity

        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _asarray(self, obj: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj)
        return obj


class MyKNNReg:

    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform") -> None:
        self.k: int = k
        self.metric: str = metric
        self.weight: str = weight
        self.train_size: Union[tuple, None] = None
        self.train_x: Union[np.ndarray, None] = None
        self.train_y: Union[np.ndarray, None] = None

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key,
                           value in self.__dict__.items())
        return f"MyKNNReg class: {params}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_x = self._asarray(X)
        self.train_y = self._asarray(y)
        self.train_size = self.train_x.shape

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = self._asarray(X)
        distances = self._pairwise_distance(X, self.train_x)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_targets = self.train_y[k_nearest_indices]

        if self.weight == "uniform":
            return np.mean(k_nearest_targets, axis=1)

        elif self.weight == "rank":
            ranks = np.arange(1, self.k + 1)[::-1]
            weighted_sum = np.sum(k_nearest_targets * ranks, axis=1)
            weight_sum = np.sum(ranks, axis=0)  # Same for all samples
            return weighted_sum / weight_sum

        elif self.weight == "distance":
            distances = distances[np.arange(
                X.shape[0])[:, None], k_nearest_indices]
            weights = 1 / (distances + 1e-5)
            weighted_sum = np.sum(k_nearest_targets * weights, axis=1)
            return weighted_sum / np.sum(weights, axis=1)

        else:
            raise ValueError(f"Invalid weight: {self.weight}")

    def _asarray(self, obj: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj)
        return obj

    def _pairwise_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=2))

        elif self.metric == "chebyshev":
            return np.max(np.abs(x[:, np.newaxis] - y), axis=2)

        elif self.metric == "manhattan":
            return np.sum(np.abs(x[:, np.newaxis] - y), axis=2)

        elif self.metric == "cosine":
            x_norm = np.linalg.norm(x, axis=1)
            y_norm = np.linalg.norm(y, axis=1)
            similarity = np.dot(x, y.T) / (x_norm[:, np.newaxis] * y_norm)
            return 1 - similarity

        else:
            raise ValueError(f"Invalid metric: {self.metric}")

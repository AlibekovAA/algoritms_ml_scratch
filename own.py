from typing import Optional, Callable, Union
import numpy as np
import pandas as pd
import random


class MyLineReg:
    def __init__(self, n_iter: int = 10, learning_rate: Union[float, Callable[[int], float]] = 0.5, weights=None, metric: Optional[str] = None,
                 reg: Optional[str] = None, l1_coef: float = 0.0, l2_coef: float = 0.0, sgd_sample: Optional[Union[int, float]] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        params = ', '.join(f'{key}={value}' for key,
                           value in self.__dict__.items())
        return f"MyLineReg class: {params}"

    def _calculate_metric(self, y_true, y_pred):
        """Расчет метрики на основе параметра metric."""
        if self.metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif self.metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif self.metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.metric == "mape":
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif self.metric == "r2":
            ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_residual = np.sum((y_true - y_pred) ** 2)
            return 1 - (ss_residual / ss_total)
        return None

    def _regularization_loss(self):
        """Вычисление потерь для регуляризации."""
        if self.reg == "l1":
            return self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == "l2":
            return self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == "elasticnet":
            return (self.l1_coef * np.sum(np.abs(self.weights))) + (self.l2_coef * np.sum(self.weights ** 2))
        return 0

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        observation_count, feature_count = X.shape
        X = np.c_[np.ones(observation_count), X]
        self.weights = np.ones(feature_count + 1)

        random.seed(self.random_state)

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample:
                if isinstance(self.sgd_sample, int):
                    sample_size = self.sgd_sample
                else:
                    sample_size = int(
                        round(self.sgd_sample * observation_count))

                sample_rows_idx = random.sample(
                    range(observation_count), sample_size)
                X_batch = X[sample_rows_idx]
                y_batch = y.iloc[sample_rows_idx]
            else:
                X_batch = X
                y_batch = y

            y_hat = X_batch.dot(self.weights)
            error = y_hat - y_batch
            mse = (error ** 2).mean()

            reg_loss = self._regularization_loss()
            loss = mse + reg_loss

            metric_value = self._calculate_metric(y_batch, y_hat)

            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate

            if verbose and (i % verbose == 0 or i == 1):
                if self.metric:
                    print(f"Iteration {i} | learning_rate: {lr:.5f} | loss: {
                          loss:.2f} | {self.metric}: {metric_value:.2f}")
                else:
                    print(f"Iteration {i} | learning_rate: {
                          lr:.5f} | loss: {loss:.2f}")

            gradient = -2 / X_batch.shape[0] * X_batch.T.dot(y_batch - y_hat)

            if self.reg == "l1":
                gradient += self.l1_coef * np.sign(self.weights)
            elif self.reg == "l2":
                gradient += 2 * self.l2_coef * self.weights
            elif self.reg == "elasticnet":
                gradient += self.l1_coef * \
                    np.sign(self.weights) + 2 * self.l2_coef * self.weights

            self.weights -= lr * gradient

        final_y_hat = X.dot(self.weights)
        self.best_score = self._calculate_metric(y, final_y_hat)

        if verbose:
            if self.metric:
                print(f"Final | loss: {loss:.2f} | {
                      self.metric}: {self.best_score:.2f}")
            else:
                print(f"Final | loss: {loss:.2f}")

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        observation_count = X.shape[0]
        X = np.c_[np.ones(observation_count), X]
        return X.dot(self.weights)

    def get_best_score(self):
        return self.best_score

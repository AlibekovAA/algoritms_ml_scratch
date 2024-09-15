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


class MyLogReg:
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable[[int], float]] = 0.1,
                 reg: Optional[str] = None, l1_coef: float = 0.0, l2_coef: float = 0.0,
                 metric: Optional[str] = None, sgd_sample: Optional[Union[int, float]] = None,
                 random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.metric = metric
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.weights = None
        self.eps = 1e-15
        self.best_score = None

    def __repr__(self):
        params = ', '.join(f'{key}={value}' for key,
                           value in self.__dict__.items())
        return f"MyLogReg class: {params}"

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __log_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + self.eps) + (1 - y_true) * np.log(1 - y_pred + self.eps))

    def __add_regularization(self):
        """Добавление регуляризации в градиенты."""
        if self.reg == "l1":
            return self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            return self.l2_coef * self.weights
        elif self.reg == "elasticnet":
            return self.l1_coef * np.sign(self.weights) + self.l2_coef * self.weights
        return np.zeros_like(self.weights)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Optional[Union[int, bool]] = False):
        X = pd.concat(
            [pd.Series(np.ones(X.shape[0]), name='intercept'), X], axis=1)
        self.weights = np.ones(X.shape[1])

        random.seed(self.random_state)

        for i in range(self.n_iter):
            if self.sgd_sample:
                if isinstance(self.sgd_sample, int):
                    sample_size = min(self.sgd_sample, X.shape[0])
                elif isinstance(self.sgd_sample, float):
                    sample_size = int(round(self.sgd_sample * X.shape[0]))
                else:
                    raise ValueError(
                        "sgd_sample should be either an int or a float between 0.0 and 1.0")

                sample_size = min(sample_size, X.shape[0])
                sample_rows_idx = np.random.choice(
                    X.shape[0], sample_size, replace=False)
                X_sample = X.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
            else:
                X_sample = X
                y_sample = y

            y_pred_proba = self.__sigmoid(np.dot(X_sample, self.weights))
            loss = self.__log_loss(y_sample, y_pred_proba)

            gradient = np.dot(X_sample.T, (y_pred_proba - y_sample)) / \
                len(y_sample) + self.__add_regularization()

            if callable(self.learning_rate):
                lr = self.learning_rate(i + 1)
            else:
                lr = self.learning_rate

            self.weights -= lr * gradient

            if verbose and (i + 1) % verbose == 0:
                y_pred_proba_full = self.__sigmoid(np.dot(X, self.weights))
                metric_value = self.__calculate_metric(
                    y, y_pred_proba_full) if self.metric else None
                if self.metric:
                    print(f"{i + 1} | loss: {loss:.2f} | learning_rate: {
                          lr:.4f} | {self.metric}: {metric_value:.2f}")
                else:
                    print(f"{i + 1} | loss: {loss:.2f} | learning_rate: {lr:.4f}")

        if verbose and not isinstance(verbose, bool):
            initial_loss = self.__log_loss(
                y, self.__sigmoid(np.dot(X, np.ones(X.shape[1]))))
            print(f"start | loss: {initial_loss:.2f}")

        final_predictions_proba = self.__sigmoid(np.dot(X, self.weights))
        if self.metric:
            self.best_score = self.__calculate_metric(
                y, final_predictions_proba)

    def __calculate_metric(self, y_true, y_pred_proba):
        y_pred = (y_pred_proba > 0.5).astype(int)
        if self.metric == 'accuracy':
            return self.__accuracy(y_true, y_pred)
        elif self.metric == 'precision':
            return self.__precision(y_true, y_pred)
        elif self.metric == 'recall':
            return self.__recall(y_true, y_pred)
        elif self.metric == 'f1':
            return self.__f1_score(y_true, y_pred)
        elif self.metric == 'roc_auc':
            return self.__roc_auc(y_true, y_pred_proba)
        else:
            raise ValueError("Unknown metric")

    def __accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def __precision(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def __recall(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def __f1_score(self, y_true, y_pred):
        precision = self.__precision(y_true, y_pred)
        recall = self.__recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def __roc_auc(self, y_true, y_pred_proba):
        thresholds = np.concatenate(
            ([0], np.sort(np.unique(y_pred_proba)), [1]))
        tpr = []
        fpr = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tpr.append(self.__recall(y_true, y_pred))
            fpr.append(self.__fpr(y_true, y_pred))

        tpr = np.array(tpr)
        fpr = np.array(fpr)
        return -np.trapz(tpr, fpr)

    def __fpr(self, y_true, y_pred):
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X: pd.DataFrame):
        observation_count = X.shape[0]
        X = np.c_[np.ones(observation_count), X]
        return self.__sigmoid(X.dot(self.weights))

    def predict(self, X: pd.DataFrame):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_best_score(self):
        return self.best_score

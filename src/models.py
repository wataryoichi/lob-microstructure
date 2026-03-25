"""Model implementations for LOB price direction prediction.

Models from Wang (2025):
- Logistic Regression
- XGBoost
- CatBoost
- DeepLOB (Conv2D + LSTM)
- CNN + LSTM (simpler variant)
- CNN + XGBoost
- CNN + CatBoost
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all prediction models."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Train and predict, returning predictions, probabilities, and training time."""
        start = time.time()
        self.fit(X_train, y_train, X_val, y_val, sample_weight)
        elapsed = time.time() - start
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)
        return preds, probs, elapsed


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline."""

    def __init__(self, seed: int = 42, **kwargs):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            solver="lbfgs",
            **kwargs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost classifier."""

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        early_stopping_rounds: int = 50,
        **kwargs,
    ):
        from xgboost import XGBClassifier
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
            verbosity=0,
            **kwargs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            self.model.set_params(objective="binary:logistic", eval_metric="logloss")
        else:
            self.model.set_params(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)

        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model.fit(X_train, y_train, **fit_params)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel):
    """CatBoost classifier."""

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        **kwargs,
    ):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=max_depth,
            random_seed=seed,
            auto_class_weights="Balanced",
            verbose=0,
            **kwargs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        self.model.fit(X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X).flatten().astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _get_torch():
    """Lazy import torch."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required for deep learning models. Install with: pip install torch"
        )


class DeepLOBModel(BaseModel):
    """DeepLOB: Conv2D blocks + LSTM.

    Architecture (from paper):
    - 3x Conv2D(1x3) + LeakyReLU + BatchNorm
    - LSTM(64)
    - Focal loss with class weights
    """

    def __init__(
        self,
        n_features: int = 40,
        n_classes: int = 2,
        seq_len: int = 1,
        hidden_dim: int = 64,
        seed: int = 42,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 256,
        **kwargs,
    ):
        torch = _get_torch()
        torch.manual_seed(seed)
        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()

    def _build_model(self):
        torch = _get_torch()
        import torch.nn as nn

        class DeepLOBNet(nn.Module):
            def __init__(self, n_feat, n_cls, hidden, seq_len):
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(16),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1)),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32),
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(64),
                )
                self.lstm = nn.LSTM(
                    input_size=64 * n_feat,
                    hidden_size=hidden,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden, n_cls)

            def forward(self, x):
                # x: (batch, seq_len, n_feat)
                batch, seq, feat = x.shape
                # Reshape to (batch*seq, 1, 1, feat) for conv
                x = x.reshape(batch * seq, 1, 1, feat)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                # Flatten conv output
                x = x.reshape(batch, seq, -1)
                # LSTM
                out, _ = self.lstm(x)
                # Take last timestep
                out = out[:, -1, :]
                return self.fc(out)

        return DeepLOBNet(self.n_features, self.n_classes, self.hidden_dim, self.seq_len).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        torch = _get_torch()
        import torch.nn as nn

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(
            weight=self._compute_class_weights_tensor(y_train)
        )

        X_t = self._prepare_input(X_train)
        y_t = torch.LongTensor(y_train).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        torch = _get_torch()
        self.model.eval()
        with torch.no_grad():
            X_t = self._prepare_input(X)
            out = self.model(X_t)
            return out.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        torch = _get_torch()
        self.model.eval()
        with torch.no_grad():
            X_t = self._prepare_input(X)
            out = self.model(X_t)
            return torch.softmax(out, dim=1).cpu().numpy()

    def _prepare_input(self, X):
        torch = _get_torch()
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # (batch, 1, feat)
        return torch.FloatTensor(X).to(self.device)

    def _compute_class_weights_tensor(self, y):
        torch = _get_torch()
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts.astype(float))
        w = np.ones(self.n_classes)
        for cls, wt in zip(classes, weights):
            w[int(cls)] = wt
        return torch.FloatTensor(w).to(self.device)


class CNNLSTMModel(BaseModel):
    """Simpler CNN + BiLSTM variant.

    - Single Conv2D layer
    - BatchNorm + SpatialDropout
    - Bidirectional LSTM(32)
    """

    def __init__(
        self,
        n_features: int = 40,
        n_classes: int = 2,
        seq_len: int = 1,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        seed: int = 42,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 256,
        **kwargs,
    ):
        torch = _get_torch()
        torch.manual_seed(seed)
        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()

    def _build_model(self):
        torch = _get_torch()
        import torch.nn as nn

        class CNNBiLSTM(nn.Module):
            def __init__(self, n_feat, n_cls, hidden, dropout):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32),
                    nn.Dropout2d(dropout),
                )
                self.lstm = nn.LSTM(
                    input_size=32 * n_feat,
                    hidden_size=hidden,
                    batch_first=True,
                    bidirectional=True,
                )
                self.fc = nn.Linear(hidden * 2, n_cls)

            def forward(self, x):
                batch, seq, feat = x.shape
                x = x.reshape(batch * seq, 1, 1, feat)
                x = self.conv(x)
                x = x.reshape(batch, seq, -1)
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        return CNNBiLSTM(self.n_features, self.n_classes, self.hidden_dim, self.dropout).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        torch = _get_torch()
        import torch.nn as nn

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        torch = _get_torch()
        self.model.eval()
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return out.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        torch = _get_torch()
        self.model.eval()
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return torch.softmax(out, dim=1).cpu().numpy()


def get_model(
    model_type: str,
    n_features: int = 40,
    n_classes: int = 2,
    seed: int = 42,
    **kwargs,
) -> BaseModel:
    """Factory function for model creation."""
    models = {
        "logistic_regression": lambda: LogisticRegressionModel(seed=seed, **kwargs),
        "xgboost": lambda: XGBoostModel(seed=seed, **kwargs),
        "catboost": lambda: CatBoostModel(seed=seed, **kwargs),
        "deeplob": lambda: DeepLOBModel(n_features=n_features, n_classes=n_classes, seed=seed, **kwargs),
        "cnn_lstm": lambda: CNNLSTMModel(n_features=n_features, n_classes=n_classes, seed=seed, **kwargs),
    }

    factory = models.get(model_type)
    if factory is None:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models)}")
    return factory()

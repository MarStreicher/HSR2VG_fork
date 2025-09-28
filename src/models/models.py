from adapt.feature_based import CORAL, SA, TCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import numpy as np
from typing import Tuple
from base_experiment import BaseExperimentArgs


class CORALModelArgs(BaseExperimentArgs):
    model: str = "CORAL"
    lambda_value: float = 0.01
    n_components: int = 2


class CORALModel:
    def __init__(self, config: CORALModelArgs):
        self.lambda_value = config.lambda_value
        self.n_components = config.n_components

    def train(self, X_source, y_source, X_target):
        self.estimator = PLSRegression(self.n_components)

        self.model = CORAL(
            estimator=self.estimator,
            Xt=X_target,
            random_state=42,
            lambda_=self.lambda_value,
            verbose=3,
            copy=False,
        )

        self.model = self.model.fit(X_source, y_source)
        return

    def validate(self, domain: str, X, y) -> Tuple[float, float]:
        preds = self.model.predict(X, domain)

        r2 = r2_score(y, preds)
        r = np.corrcoef(preds, y)[0, 1]
        return r2, r

    @classmethod
    def get_args_model(cls):
        return CORALModelArgs


class SAModelArgs(BaseExperimentArgs):
    model: str = "SA"
    n_components: int = 2
    sa_n_component: int = 2


class SAModel:
    def __init__(self, config: SAModelArgs):
        self.sa_n_component = config.sa_n_component
        self.n_components = config.n_components

    def train(self, X_source, y_source, X_target):
        self.estimator = PLSRegression(self.n_components)

        self.model = SA(
            estimator=self.estimator,
            Xt=X_target,
            random_state=42,
            n_components=self.sa_n_component,
            verbose=3,
            copy=False,
        )

        self.model = self.model.fit(X_source, y_source)
        return

    def validate(self, domain: str, X, y) -> Tuple[float, float]:
        preds = self.model.predict(X, domain)

        r2 = r2_score(y, preds)
        r = np.corrcoef(preds, y)[0, 1]
        return r2, r

    @classmethod
    def get_args_model(cls):
        return SAModelArgs


class TCAModelArgs(BaseExperimentArgs):
    model: str = "TCA"
    n_components: int = 2
    mu: float = 2.0
    kernel: str = "linear"


class TCAModel:
    def __init__(self, config: TCAModelArgs):
        self.mu = config.mu
        self.n_components = config.n_components
        self.kernel = config.kernel

    def train(self, X_source, y_source, X_target):
        self.estimator = PLSRegression(self.n_components)

        self.model = TCA(
            estimator=self.estimator,
            Xt=X_target,
            random_state=42,
            mu=self.mu,
            kernel=self.kernel,
            verbose=3,
            copy=False,
        )

        self.model = self.model.fit(X_source, y_source)
        return

    def validate(self, domain: str, X, y) -> Tuple[float, float]:
        preds = self.model.predict(X, domain)

        r2 = r2_score(y, preds)
        r = np.corrcoef(preds, y)[0, 1]
        return r2, r

    @classmethod
    def get_args_model(cls):
        return TCAModelArgs

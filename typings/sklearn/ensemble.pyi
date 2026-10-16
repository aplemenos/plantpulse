from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from sklearn._typing import FloatArray, MatrixLike, VectorLike
from sklearn.base import BaseEstimator

class GradientBoostingRegressor(BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int = ...,
        max_depth: int | None = ...,
        learning_rate: float = ...,
        random_state: int | None = ...,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: VectorLike) -> Self: ...
    def predict(self, X: MatrixLike) -> FloatArray: ...

class RandomForestClassifier(BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int = ...,
        max_depth: int | None = ...,
        min_samples_leaf: int = ...,
        class_weight: Literal["balanced", "balanced_subsample"] | dict[int, float] | None = ...,
        random_state: int | None = ...,
        n_jobs: int | None = ...,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: VectorLike) -> Self: ...
    def predict(self, X: MatrixLike) -> NDArray[np.int64]: ...
    def predict_proba(self, X: MatrixLike) -> FloatArray: ...

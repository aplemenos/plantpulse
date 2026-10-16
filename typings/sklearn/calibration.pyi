from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from sklearn._typing import FloatArray, MatrixLike, VectorLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class _CalibratedClassifier:
    estimator: RandomForestClassifier

class CalibratedClassifierCV(BaseEstimator):
    calibrated_classifiers_: list[_CalibratedClassifier]
    def __init__(
        self,
        estimator: RandomForestClassifier | None = ...,
        *,
        method: Literal["sigmoid", "isotonic"] = ...,
        cv: int | None = ...,
        ensemble: bool = ...,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: VectorLike) -> Self: ...
    def predict(self, X: MatrixLike) -> NDArray[np.int64]: ...
    def predict_proba(self, X: MatrixLike) -> FloatArray: ...

def calibration_curve(
    y_true: VectorLike,
    y_prob: VectorLike,
    *,
    n_bins: int = ...,
    strategy: Literal["uniform", "quantile"] = ...,
) -> tuple[FloatArray, FloatArray]: ...

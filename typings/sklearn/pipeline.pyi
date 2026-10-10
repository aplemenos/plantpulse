from typing import Self

from sklearn._typing import FloatArray
from sklearn.base import BaseEstimator

class Pipeline(BaseEstimator):
    def fit(self, X: list[str]) -> Self: ...
    def transform(self, X: list[str]) -> FloatArray: ...

def make_pipeline(*steps: BaseEstimator) -> Pipeline: ...

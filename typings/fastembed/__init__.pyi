from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

class TextEmbedding:
    def __init__(self, model_name: str = ...) -> None: ...
    def embed(self, documents: list[str]) -> Iterable[NDArray[np.float32]]: ...

# Minimal local stubs covering only the scikit-learn API PlantPulse uses.
# scikit-learn ships no type information, so these keep pyright strict mode
# meaningful at the library boundary instead of letting Unknown types spread.
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MatrixLike: TypeAlias = pd.DataFrame | NDArray[Any]
VectorLike: TypeAlias = pd.Series[Any] | NDArray[Any]
FloatArray: TypeAlias = NDArray[np.float64]

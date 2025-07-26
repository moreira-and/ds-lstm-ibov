from config import logger
from ..interfaces import ICleanStrategy

import numpy as np
import pandas as pd

class PercentageReturnCalculator(ICleanStrategy):
    """
    Cleaning strategy that calculates percentage returns (rate of change) for all features.

    Operations:
    1. Applies .pct_change to all columns.
    2. Replaces infinite values with 0.
    3. Fills any remaining NaNs with 0.
    """
    def __init__(self, periods: int = 1):
        self.periods = periods

    def clear(self, X: pd.DataFrame, y=None):
        combined = pd.concat([X, y], axis=1) if y is not None else X.copy()

        combined = combined.pct_change(periods=self.periods, fill_method=None)
        combined = combined.replace([np.inf, -np.inf], 0).fillna(0)

        if y is not None:
            X_cleaned = combined[X.columns]
            y_cleaned = combined[y.name] if isinstance(y, pd.Series) else combined[y.columns]
            return X_cleaned, y_cleaned
        else:
            return combined, y
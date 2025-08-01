from config import logger
from ..interfaces import ICleanStrategy

import pandas as pd
from typing import List, Optional

class MissingValueFiller(ICleanStrategy):
    """
    Cleaning strategy that handles missing values in the dataset.

    Operations:
    1. Sorts the dataframe by index.
    2. Optionally drops rows with NaNs in the specified target columns.
    3. Applies forward-fill followed by backward-fill to remaining missing values.

    Parameters:
    ----------
    targets : Optional[List[str]]
        List of target column names to check for initial missing values.
        If None, no row-dropping is performed.
    """

    def __init__(self, targets: Optional[List[str]] = None):
        self.targets = targets

    def clear(self, X: pd.DataFrame, y:pd.DataFrame =None):

        X.sort_index(inplace=True)

        if y is not None:
            y.sort_index(inplace=True)
            assert (X.index == y.index).all(), "X and y indices do not match after sorting"

        combined = pd.concat([X, y], axis=1) if y is not None else X.copy()

        if self.targets is not None:
            combined = combined.dropna(subset=self.targets)

        combined = combined.ffill().bfill()

        if y is not None:
            X_cleaned = combined[X.columns]
            y_cleaned = combined[y.name] if isinstance(y, pd.Series) else combined[y.columns]
            return X_cleaned, y_cleaned

        return combined, y
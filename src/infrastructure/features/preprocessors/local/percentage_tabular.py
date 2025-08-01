from ..interfaces import IPreprocessorStrategy
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

class PercentageTabularPreprocessor(IPreprocessorStrategy):
    """
    Preprocessor that calculates percentage returns (rate of change) for all features.

    Operations:
    1. Applies .pct_change to all columns.
    2. Replaces infinite values with 1 or -1.
    3. Fills NaNs with 0.
    4. Inverse_transform rebuilds original data from percentage returns using stored original values.
    """

    def __init__(self, periods: int = 1):
        self.periods = periods
        self._original_X = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        # Store original X for inverse_transform
        if isinstance(X, np.ndarray):
            self._original_X = pd.DataFrame(X)
        else:
            self._original_X = X.copy()

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:

        if isinstance(X, pd.DataFrame):
            X = X.sort_index()
        if y is not None and hasattr(y, 'sort_index'):
            y = y.sort_index()
            assert (X.index == y.index).all(), "X and y indices do not match after sorting"

        combined = pd.concat([X, y], axis=1) if y is not None else X.copy()

        combined = combined.pct_change(periods=self.periods, fill_method="ffill")
        combined = combined.replace(np.inf, 1).replace(-np.inf, -1).fillna(0)

        if y is not None:
            X_transformed = combined[X.columns]
            y_transformed = combined[y.name] if isinstance(y, pd.Series) else combined[y.columns]
            return X_transformed, y_transformed

        return combined, y

    def inverse_transform(
        self,
        X_pct: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:

        if self._original_X is None:
            raise RuntimeError("Must call fit() before inverse_transform()")

        if isinstance(X_pct, np.ndarray):
            X_pct = pd.DataFrame(X_pct, columns=self._original_X.columns, index=self._original_X.index)
        
        # Rebuild original values from percentage returns:
        # Formula: original_t = original_{t-1} * (1 + pct_change_t)
        X_inv = self._original_X.copy()
        for col in X_pct.columns:
            for i in range(self.periods, len(X_pct)):
                X_inv.iloc[i, X_inv.columns.get_loc(col)] = (
                    X_inv.iloc[i - self.periods, X_inv.columns.get_loc(col)] * (1 + X_pct.iloc[i, X_pct.columns.get_loc(col)])
                )
        # For rows before periods, keep original (or could be nan)

        if y is not None:
            if isinstance(y, np.ndarray):
                y = pd.DataFrame(y, columns=self._original_X.columns, index=self._original_X.index)
            return X_inv, y

        return X_inv, y

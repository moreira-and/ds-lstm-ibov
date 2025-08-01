"""
features_pipeline.py

This module defines the `FeaturesPipeline` class, which orchestrates a modular and sequential execution
of key feature engineering steps: cleaning, splitting, selection, and preprocessing. Each step is
represented by a dedicated pipeline or strategy adhering to a consistent interface contract, enabling
flexibility, extensibility, and reuse.

### Interfaces Used:
- ISplitterStrategy: Interface for data splitting logic (e.g., train-test split, time-based split).
- CleanPipeline: Composite pipeline of ICleanStrategy implementations applied sequentially.
- SelectPipeline: Composite pipeline of ISelectStrategy implementations, supporting fitting and selection.
- PreprocessorPipeline: Composite pipeline of IPreprocessorStrategy implementations, supporting fitting and transformation.

### Typical usage:
```python
pipeline = FeaturesPipeline(
    splitter=SequentialLengthSplitter(),
    clean_pipeline=CleanPipeline([...]),
    select_pipeline=SelectPipeline([...]),
    preprocess_pipeline=PreprocessorPipeline([...])
)

X_train, y_train, X_test, y_test, artifacts = pipeline.run(X_raw, y_raw)
"""

from splitters.interfaces import ISplitterStrategy
from cleaners.runners import CleanPipeline
from selectors.runners import SelectPipeline
from preprocessors.runners import PreprocessorPipeline

class FeaturesPipeline:
    """
    Orchestrates modular feature engineering steps:
    - Clean
    - Split
    - Select
    - Preprocess
    """

    def __init__(
        self,
        preprocess_pipeline: PreprocessorPipeline,
        clean_pipeline: CleanPipeline = None,
        splitter: ISplitterStrategy = None,
        select_pipeline: SelectPipeline = None
    ):
        self.preprocess_pipeline = preprocess_pipeline
        self.clean_pipeline = clean_pipeline
        self.splitter = splitter
        self.select_pipeline = select_pipeline

    def run(self, X, y=None):
        """
        Executes the pipelines in sequence.

        Returns:
            Tuple:
                - X_train
                - y_train
                - X_test
                - y_test
                - artifacts: dict with keys:
                    - "clean"
                    - "split"
                    - "select"
                    - "preprocess"
        """
        artifacts = {}

        # Clean
        if self.clean_pipeline is not None:
            X, y, clean_artifact = self.clean_pipeline.clear(X, y)
            artifacts["clean"] = clean_artifact

        # Split
        if self.splitter is not None:
            X_train, X_test, y_train, y_test, split_artifact = self.splitter.split(X, y)
            artifacts["split"] = split_artifact
        else:
            X_train, y_train = X, y
            X_test = y_test = None

        # Select
        if self.select_pipeline is not None:
            X_train, y_train, select_artifact = self.select_pipeline.fit_select(X_train, y_train)
            artifacts["select"] = select_artifact

        # Preprocess (train)
        X_train, y_train, preprocess_artifact = self.preprocess_pipeline.fit_transform(X_train, y_train)
        artifacts["preprocess"] = preprocess_artifact

        # Preprocess (test)
        if X_test is not None and y_test is not None:
            if self.select_pipeline is not None:
                X_test, y_test = self.select_pipeline.select(X_test, y_test)
            X_test, y_test = self.preprocess_pipeline.transform(X_test, y_test)

        return X_train, y_train, X_test, y_test, artifacts

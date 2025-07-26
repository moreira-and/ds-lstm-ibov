from config import logger
from .interfaces import (
    IPrepareDataTemplate,
    ISplitterStrategy,
    ITransformStrategy,
    IGeneratorStrategy
)
from .processors.default_rnn_processors import DefaultRnnPreprocessor

import numpy as np

class DefaultRnnPrepareData(IPrepareDataTemplate):
    """
    Template implementation for preparing data for RNN models using a modular strategy pattern.

    This class orchestrates the pipeline to:
    1. Split the input dataset into train and test sets using a splitter strategy.
    2. Transform the datasets using a transformation strategy (scaling, encoding, etc.).
    3. Generate sequences/windows using a generator strategy.
    4. Separate the generated data into X_train, X_test, y_train, y_test.

    Attributes
    ----------
    dataset : pd.DataFrame
        The original input dataset (features + targets).
    targets : List[str] or None
        Column names to be used as targets (optional).
    splitter : ISplitterStrategy
        Strategy object responsible for splitting the dataset.
    transformer : ITransformStrategy
        Strategy object for transforming the dataset.
    generator : IGeneratorStrategy
        Strategy object that generates sequences or windows from the data.
    """

    def __init__(self, dataset, targets, splitter: ISplitterStrategy,
                 transformer: ITransformStrategy, generator: IGeneratorStrategy):
        self.dataset = dataset
        self.targets = targets
        self.splitter = splitter
        self.transformer = transformer
        self.generator = generator

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def prepare_data(self):
        """
        Executes the data preparation pipeline:
        - Split
        - Transform
        - Generate sequences
        - Separate into train/test datasets
        """
        logger.info("Starting data preparation pipeline.")

        logger.debug("Splitting dataset.")
        train_data, test_data = self.splitter.split(X=self.dataset)

        logger.debug("Fitting transformer on training data.")
        train_X = self.transformer.fit_transform(X=train_data)

        if test_data is None or len(test_data) == 0:
            logger.warning("No test data provided. Will use only train data for generation.")
            test_X = None
            X_all = train_X
        else:
            logger.debug("Transforming test data.")
            test_X = self.transformer.transform(X=test_data)
            logger.debug("Concatenating transformed train and test data for generator.")
            X_all = np.concatenate([train_X[0], test_X[0]])

        if self.targets is None or len(self.targets) == 0:
            logger.warning("No target columns specified. Generator will use all data.")
            generator = self.generator.generate(data=X_all)
        else:
            logger.debug("Identifying target features by name matching.")
            feature_names = self.transformer.get_feature_names()
            y_features = [
                (i, feature)
                for i, feature in enumerate(feature_names)
                if any(t in feature for t in self.targets)
            ]
            y_index = [i for i, _ in y_features]
            logger.debug(f"Target feature indices identified: {y_index}")
            generator = self.generator.generate(data=X_all, targets=X_all[:, y_index])

        n_test = len(test_data)
        n_total = len(generator)
        train_end = n_total - n_test

        logger.debug(f"Splitting generated sequences: {train_end} train / {n_test} test")
        self._X_train = np.array([generator[i][0][0] for i in range(train_end)])
        self._y_train = np.array([generator[i][1][0] for i in range(train_end)])
        self._X_test = np.array([generator[i][0][0] for i in range(train_end, n_total)])
        self._y_test = np.array([generator[i][1][0] for i in range(train_end, n_total)])

        logger.info("Data preparation pipeline completed.")

    def get_data(self):
        """
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        return self._X_train, self._X_test, self._y_train, self._y_test

    def get_preprocessor(self):
        """
        Returns a preprocessor object that can be used to inverse transform or serialize processing.

        Returns:
        --------
        DefaultRnnPreprocessor
        """
        return DefaultRnnPreprocessor(self.transformer, self.generator)

    def get_postprocessor(self):
        """
        Returns a postprocessor to convert model outputs back to the original scale or domain.

        Returns:
        --------
        object
            Postprocessor object compatible with the transformer's logic.
        """
        X, _ = self.splitter.split(X=self.dataset)

        if self.targets is None or len(self.targets) == 0:
            filtered_columns = X.columns.tolist()
        else:
            filtered_columns = [
                feature for feature in X
                if any(t in feature for t in self.targets)
            ]

        logger.debug(f"Creating postprocessor for columns: {filtered_columns}")
        return self.transformer.get_postprocessor(X.loc[:, filtered_columns])

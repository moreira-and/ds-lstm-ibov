from abc import ABC, abstractmethod

class IModelBuilder(ABC):
    @abstractmethod
    def build_model(self):
        pass



class ICompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class IMetricStrategy(ABC):
    @abstractmethod
    def get_metrics(self):
        pass

class ITrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass

class IGeneratorStrategy(ABC):
    @abstractmethod
    def generate(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        targets: Optional[list[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Defines an interface for generating new data or features.
        """
        pass
from abc import ABC, abstractmethod

class IPredictTemplate(ABC):
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("Implement in subclass")

class DefaultPredictTemplate(IPredictTemplate):
    def __init__(self, model, preprocessor = None , postprocessor = None):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor
        
    def predict(self, X):
        clean_X = self.preprocessor.transform(X)
        raw_y = self.model.predict(clean_X)
        return self.postprocessor.transform(raw_y)

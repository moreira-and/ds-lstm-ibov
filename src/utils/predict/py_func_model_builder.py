from abc import ABC, abstractmethod
import mlflow.pyfunc

class PyFuncModelBuilder(ABC):
    @abstractmethod
    def build(self) -> mlflow.pyfunc.PythonModel:
        """Constrói e retorna um pyfunc model"""
        pass

class DefaultFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model, preprocessor=None, postprocessor=None):
        self.trained_model = trained_model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def predict(self, context, model_input):
        X = self.preprocessor.transform(model_input) if self.preprocessor else model_input
        preds = self.trained_model.predict(X)
        return self.postprocessor.transform(preds) if self.postprocessor else pd.DataFrame(preds)

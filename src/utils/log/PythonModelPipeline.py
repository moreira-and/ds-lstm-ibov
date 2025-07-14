
import numpy as np
import pandas as pd
import pickle
import mlflow.pyfunc



class PythonModelPipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(context.artifacts["postprocessor"], "rb") as f:
            self.postprocessor = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor.transform(model_input)
        y_pred = self.model.predict(X_transformed)
        return self.postprocessor.transform(y_pred)
import mlflow.pyfunc
import cloudpickle

# Abstract base class for PyFunc models using pre/post-processing
class PyFuncModelTemplate(mlflow.pyfunc.PythonModel):
    def __init__(self, model, preprocessor=None, postprocessor=None):
        """
        Initializes the PyFunc model components.
        
        Args:
            model: Trained model object.
            preprocessor: Preprocessing object (e.g., transformer or scaler).
            postprocessor: Postprocessing object (e.g., inverse transformer).
        """
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def load_context(self, context):
        """
        Must be implemented by subclasses. Loads artifacts from the context provided by MLflow.
        """
        raise NotImplementedError("Subclasses must implement 'load_context'.")

    def predict(self, context, model_input, params=None):
        """
        Must be implemented by subclasses. Performs inference using model and processors.
        """
        raise NotImplementedError("Subclasses must implement 'predict'.")


# Concrete implementation for LSTM model with preprocessing and postprocessing
class PyFuncModelLSTM(PyFuncModelTemplate):
    def load_context(self, context):
        """
        Loads serialized artifacts (model, preprocessor, postprocessor) from the MLflow context.
        
        Args:
            context: PythonModelContext containing artifact URIs.
        """

        with open(context.artifacts["model"], "rb") as f:
            self.model = cloudpickle.load(f)
        
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.preprocessor = cloudpickle.load(f)

        with open(context.artifacts["postprocessor"], "rb") as f:
            self.postprocessor = cloudpickle.load(f)


    def predict(self, context, model_input, params=None):
        """
        Applies preprocessing, runs model inference, and applies postprocessing.
        
        Args:
            context: Provided by MLflow during inference (not used here).
            model_input: Input data (e.g., pandas DataFrame or ndarray).
            params: Optional inference parameters.
        
        Returns:
            Postprocessed model predictions.
        """
        X = self.preprocessor.transform(model_input)
        y_pred = self.model.predict(X)
        return self.postprocessor.inverse_transform(y_pred)

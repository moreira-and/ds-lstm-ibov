from abc import ABC, abstractmethod
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow.keras.backend as K


class MetricStrategy(ABC):
    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError("Implement in subclass")

class RegressionMetricStrategy(MetricStrategy):
    def get_metrics(self):
        return ['mae', 'mse',smape,rmse,r2_score]

class ClassificationMetricStrategy(MetricStrategy):
    def get_metrics(self):
        return ['accuracy', 'precision', 'recall']
    
# Custom metrics for regression
# SMAPE
def smape(y_true, y_pred):
    numerator = K.abs(y_true - y_pred)
    denominator = (K.abs(y_true) + K.abs(y_pred)) / 2.0
    return 100.0 * K.mean(numerator / (denominator + K.epsilon()))

# RMSE
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# R² Score
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

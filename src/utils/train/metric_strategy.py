from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable

class MetricStrategy(ABC):
    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError("Implement in subclass")

class RegressionMetricStrategy(MetricStrategy):
    def get_metrics(self):
        return ['mae', 'mse',smape,rmse,r2_score]
        #return ['mae', 'mse', smape, rmse, R2Score()]

class ClassificationMetricStrategy(MetricStrategy):
    def get_metrics(self):
        return ['accuracy', 'precision', 'recall']
    
# Custom metrics for regression
# SMAPE
@register_keras_serializable()
def smape(y_true, y_pred):
    numerator = K.abs(y_true - y_pred)
    denominator = (K.abs(y_true) + K.abs(y_pred)) / 2.0
    return 100.0 * K.mean(numerator / (denominator + K.epsilon()))

# RMSE
@register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# R² Score
@register_keras_serializable()
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


@register_keras_serializable()
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name="r2_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name="ssr", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        self.ssr.assign_add(ss_res)
        self.sst.assign_add(ss_tot)
        self.count.assign_add(1.0)

    def result(self):
        return 1 - self.ssr / (self.sst + K.epsilon())

    def reset_states(self):
        self.ssr.assign(0.0)
        self.sst.assign(0.0)
        self.count.assign(0.0)

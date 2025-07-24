from train.keras.interfaces import IMetricStrategy
from train.keras.measurers.CustomMetrics import smape, rmse, r2_score

class RegressionMetrics(IMetricStrategy):
    def get_metrics(self):
        return ['mae', 'mse',smape,rmse,r2_score]
from train.interface import IMetricStrategy
from src.train.metrics_strategy.CustomMetrics import smape, rmse, r2_score, mae, mse

class RegressionMetrics(IMetricStrategy):
    def get_metrics(self):
        return ['mae', 'mse',smape,rmse,r2_score]
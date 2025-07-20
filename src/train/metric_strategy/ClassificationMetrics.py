from src.train.interfaces import IMetricStrategy

class ClassificationMetrics(IMetricStrategy):
    def get_metrics(self):
        return ['accuracy', 'precision', 'recall']
    

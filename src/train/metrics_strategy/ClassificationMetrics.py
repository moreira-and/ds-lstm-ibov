from train.interface import IMetricStrategy

class ClassificationMetrics(IMetricStrategy):
    def get_metrics(self):
        return ['accuracy', 'precision', 'recall']
    

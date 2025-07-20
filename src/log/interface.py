from abc import ABC, abstractmethod

class ILogStrategy(ABC):
    @abstractmethod
    def log(self):
        pass

class IPipelineRunner(ABC):
    @abstractmethod
    def run(self, 
            experiment_name="default_experiment",
            run_name="training_run",
            **kwargs):
        pass
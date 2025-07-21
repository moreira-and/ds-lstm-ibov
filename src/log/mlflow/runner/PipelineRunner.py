from log import ILogStrategy, IPipelineRunner
from typing import List
from mlflow import set_experiment, start_run

class PipelineRunner(IPipelineRunner):
    def __init__(self, loggers:List[ILogStrategy],):
        self.loggers = loggers


    def run(self, *,        
            experiment_name="default_experiment",
            run_name="training_run",            
            model_name="pipeline",
            **kwargs):
        """
        Executa a estratégia de log.

        Args:
            experiment_name: Nome do experimento no MLflow.
            run_name: Nome da execução.
            model_name: Nome do modelo logado.
        """
        set_experiment(experiment_name)
        logger.info(f"Starting MLflow run '{run_name}'")

        with start_run(run_name=run_name):

            logger.info("Starting log pipeline...")

            for logger in self.loggers:
                if isinstance(logger, ILogStrategy):
                    logger.info(f"Running {logger.__class__.__name__}")
                    logger.run( **kwargs)
                else:
                    raise TypeError(f"Logger {logger} does not implement ILogStrategy interface.")
                
            logger.info("Log pipeline completed.")
            
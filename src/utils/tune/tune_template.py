from abc import ABC,abstractmethod

from src.config import logger

from src.utils.tune.search_strategy import SearchStrategy
from src.utils.tune.tuner_builder import TunerBuilder

class TuneTemplate(ABC):
    @abstractmethod
    def run(self, X_train, y_train):
        pass

class TunerKerasPipeline(TuneTemplate):
    def __init__(self, tuner_builder: TunerBuilder, searcher: SearchStrategy):
        self.tuner_builder = tuner_builder
        self.searcher = searcher
        self.tuner = None

    def run(self, X_train, y_train):
        try:
            # Cria o tuner
            self.tuner = self.tuner_builder.build_tuner()

            # Executa a busca com a estratégia passada
            self.searcher.search(self.tuner, X_train, y_train)

            # Recupera os melhores hiperparâmetros
            best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

            # Constrói o melhor modelo
            best_model = self.tuner.hypermodel.build(best_hps)
            
            return best_model, best_hps

        except Exception as e:
            logger.error(f'Error running {self.__class__.__name__}: {e}')
            raise

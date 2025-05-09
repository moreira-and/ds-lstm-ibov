
from abc import ABC, abstractmethod


class PostprocessorStrategy(ABC):  
    @abstractmethod
    def inverse_transform(self, y_predicted, y_indices):
        pass


class DefaultPostprocessor(PostprocessorStrategy):

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.__get_transform_map()

    def __get_transform_map(self):

        self.transform_map = []
        output_slices = self.preprocessor.output_indices_
        
        for name, sl in output_slices.items():
            if name not in self.preprocessor.named_transformers_:
                transformer = None
            else:
                transformer = self.preprocessor.named_transformers_[name]
            self.transform_map.append((name, transformer, sl))

    def inverse_transform(self, y_predicted, y_indices):
        y_rec = y_predicted.copy()

        for name, transformer, sl in self.transform_map:
            if transformer is None or not hasattr(transformer, "inverse_transform"):
                continue

            col_range = range(sl.start, sl.stop)
            # Só tentar inverter se TODA a fatia estiver incluída nos índices
            if all(i in y_indices for i in col_range):
                part = y_predicted[:, col_range]
                inv = transformer.inverse_transform(part)
                if inv.ndim == 1:
                    inv = inv.reshape(-1, 1)
                y_rec[:, col_range] = inv
            else:
                # Se parte da fatia está fora de y_indices, pula a inversão (evita erro)
                continue

        return y_rec
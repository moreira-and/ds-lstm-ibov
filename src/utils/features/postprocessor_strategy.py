from abc import ABC, abstractmethod
import pandas as pd

class PostprocessorStrategy(ABC):  
    @abstractmethod
    def inverse_transform(self, y_predicted):
        pass

class DefaultLstmPostprocessor(PostprocessorStrategy):

    def __init__(self, transformer,y_train):
        self.__transformer = transformer.fit(y_train)


    def inverse_transform(self, y_predicted):
        # Realiza o inverse_transform normalmente
        y_inversed = self.__transformer.inverse_transform(y_predicted)

        # Verifica se há nomes de colunas disponíveis
        if hasattr(self.__transformer, 'feature_names_in_'):
            column_names = self.__transformer.feature_names_in_
        else:
            # Caso não tenha sido fit com DataFrame, usa nomes genéricos
            column_names = [f'feature_{i}' for i in range(y_inversed.shape[1])]

        return pd.DataFrame(y_inversed, columns=column_names)

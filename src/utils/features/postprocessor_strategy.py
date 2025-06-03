from abc import ABC, abstractmethod

class PostprocessorStrategy(ABC):  
    @abstractmethod
    def inverse_transform(self, y_predicted):
        pass

class DefaultLstmPostprocessor(PostprocessorStrategy):

    def __init__(self, transformer):
        self.__transformer = transformer

    def inverse_transform(self, y_predicted):
        return self.__transformer.inverse_transformer(self.__transformer, y_predicted)
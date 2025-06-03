from abc import ABC, abstractmethod

class PostprocessorStrategy(ABC):  
    @abstractmethod
    def inverse_transform(self, y_predicted):
        pass

class DefaultLstmPostprocessor(PostprocessorStrategy):

    def __init__(self, transformer,y_train):
        self.__transformer = transformer.fit(y_train)

    def inverse_transform(self, y_predicted):
        return self.__transformer.inverse_transform(y_predicted)
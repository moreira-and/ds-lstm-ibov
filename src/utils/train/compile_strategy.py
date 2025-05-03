

class CompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class ClassificationCompileStrategy(CompileStrategy):
    def compile(self, model):
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

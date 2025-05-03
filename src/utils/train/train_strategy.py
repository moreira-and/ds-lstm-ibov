class TrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass

class BasicTrainStrategy(TrainStrategy):
    def __init__(self, epochs=20, batch_size=32, validation_split=0.2, callbacks = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.callbacks = callbacks

    def train(self, model, X_train, y_train):
        return model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks = self.callbacks
        )



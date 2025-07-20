
from log.interface import ILogStrategy
from src import config

from mlflow import log_artifact

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LogValPredictionsPlot(ILogStrategy):
    def __init__(self, model, model_name, validation_len):
        self.model = model
        self.model_name = model_name
        self.validation_len = validation_len

    def run(self, **kwargs):

        X_val = np.load(config.X_PROCESSED_DATA_TRAIN_FILE)[-self.validation_len:]
        y_val = np.load(config.Y_PROCESSED_DATA_TRAIN_FILE)[-self.validation_len:]

        y_pred = self.model.predict(X_val)

        config.logger.info("Plotting model predictions...")
        postprocessor = pickle.load(open(config.MAIN_POSTPROCESSOR_FILE, "rb"))

        # y_val and y_pred should be DataFrames after inverse_transform
        y_val = postprocessor.inverse_transform(y_val)
        y_pred = postprocessor.inverse_transform(y_pred)

        # If they are still numpy arrays, convert to DataFrame with matching column names
        if not isinstance(y_val, pd.DataFrame):
            y_val = pd.DataFrame(y_val, columns=[f"Output {i+1}" for i in range(y_val.shape[1])])
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_val.columns)

        # Number of output variables
        n_outputs = y_val.shape[1]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(10, 5 * n_outputs), sharex=True)

        # Ensure axs is iterable even if there is only one plot
        if n_outputs == 1:
            axs = [axs]

        # Plot each output with proper axis labeling
        for i, col in enumerate(y_val.columns):
            axs[i].plot(y_val[col], label="True", color="blue")
            axs[i].plot(y_pred[col], label="Predicted", color="orange")
            axs[i].set_ylabel(col)
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel("Time")
        fig.suptitle("Model Predictions vs True Values on Validation Set")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = config.FIGURES_DIR / f"validation_predictions.png"
        plt.savefig(plot_path)
        log_artifact(plot_path, artifact_path="plots")
        plt.close()
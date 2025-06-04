from pathlib import Path
import pandas as pd

import cloudpickle
import keras
from loguru import logger
from tqdm import tqdm
import typer

from src import config, utils
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    preprocessor_path: Path = PROCESSED_DATA_DIR / "preprocessor.pkl",
    model_path: Path = MODELS_DIR / "Sequential_epoch59_loss0.1171.keras",    
    postprocessor_path: Path = PROCESSED_DATA_DIR / "postprocessor.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "dataset_report.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")

    # Carrega dados de entrada
    df = pd.read_csv(input_path, index_col=0,parse_dates=True)
    logger.info(f"Input data shape: {df.tail(8).shape}")
    
    with open(preprocessor_path, "rb") as f:
        preprocessor = cloudpickle.load(f)
    # Aplica pipeline de predição
    X_processed = preprocessor.transform(df.tail(8))

    model = keras.models.load_model(model_path)
    predictions = model.predict(X_processed)

    with open(postprocessor_path, "rb") as f:
        postprocessor = cloudpickle.load(f)

    df_predicted = postprocessor.inverse_transform(predictions)
    df_predicted['type'] = 'Predicted'

    last_index = df.index[-1]
    new_index = last_index + pd.Timedelta(days=1)
    df_predicted.index = [new_index]

    
    df['type'] = 'True'
    df_report = pd.concat([df, df_predicted])
    df_report = df_report.ffill()

    # Salva resultados
    df_report.to_csv(output_path, index=True)


    # Log com MLflow
    #with mlflow.start_run():
    #    mlflow.log_param("input_path", input_path)
    #    mlflow.log_artifact(output_path)
    #    mlflow.set_tag("pipeline", "predict_regression")

    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

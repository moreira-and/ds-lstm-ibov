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
    model_path: Path = MODELS_DIR / "Sequential_epoch13_loss0.0875.keras",    
    postprocessor_path: Path = PROCESSED_DATA_DIR / "postprocessor.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "y_predicted.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")

    model = keras.models.load_model(model_path)
    
    with open(preprocessor_path, "rb") as f:
        preprocessor = cloudpickle.load(f)

    with open(postprocessor_path, "rb") as f:
        postprocessor = cloudpickle.load(f)

    # Carrega dados de entrada
    df = pd.read_csv(input_path, index_col=0)
    logger.info(f"Input data shape: {df.shape}")

    # Aplica pipeline de predição
    X_processed = preprocessor.transform(df)
    predictions = model.predict(X_processed)
    final_output = postprocessor.inverse_transform(predictions)

    # Salva resultados
    pd.DataFrame(final_output, columns=["prediction"]).to_csv(output_path, index=False)


    # Log com MLflow
    #with mlflow.start_run():
    #    mlflow.log_param("input_path", input_path)
    #    mlflow.log_artifact(output_path)
    #    mlflow.set_tag("pipeline", "predict_regression")

    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

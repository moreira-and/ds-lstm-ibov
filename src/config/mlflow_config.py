from mlflow import set_tracking_uri
from .paths import PROJ_ROOT

def configure_mlflow():
    mlflow_tracking_uri = PROJ_ROOT / "mlruns"
    set_tracking_uri(mlflow_tracking_uri)

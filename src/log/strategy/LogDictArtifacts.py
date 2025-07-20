from log.interface import ILogStrategy
from src import config

from mlflow import log_artifact
from pathlib import Path

class LogDictArtifacts(ILogStrategy):
    def __init__(self, *, artifact_paths: dict[str, Path]):
        """
        Initialize the logger with a dictionary of artifacts.
        Expects a dict in the form: {"name": Path("/path/to/artifact.ext")}
        """
        self.artifact_paths = artifact_paths or {
            "preprocessor" :config.MAIN_PREPROCESSOR_FILE,
            "model" : config.MAIN_MODEL_FILE,
            "postprocessor" : config.MAIN_POSTPROCESSOR_FILE
        }

    def run(self, **kwargs):
        """
        Logs each artifact in the dictionary to MLflow.
        """
        config.logging_config.info("Logging artifacts...")

        for name, path in self.artifact_paths.items():
            if path and Path(path).is_file():
                log_artifact(local_path=path, artifact_path=f"artifacts/{name}")
                config.logger.info(f"Logged artifact: {name} → {path}")
            else:
                config.logger.warning(f"Artifact '{name}' not found or invalid: {path}")

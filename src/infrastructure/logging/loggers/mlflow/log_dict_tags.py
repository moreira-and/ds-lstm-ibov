from src.log.interfaces import ILogStrategy
from src import config

from mlflow import set_tag

import os
from src.utils.getgit_commit_hash import get_git_commit_hash

import socket
import platform

class LogDictTags(ILogStrategy):
    def __init__(self, tags_dict: dict):
        """
        Initializes the logger with a dictionary of parameters.
        {"param": "value", ...}
        """

        self.param_dict = tags_dict or {
                "framework": "keras",
                "developer": os.getenv("USER", "unknown"),
                "purpose": "regression-pipeline",
                "hostname" : socket.gethostname(),
                "git_commit": get_git_commit_hash(),
                "python_version" : platform.python_version(),
                "platform" : platform.platform(),
                "processor" : platform.processor()   
            }

    def run(self, **kwargs):        
        for tag, value in self.param_dict.items():
            if value is not None:
                set_tag(tag, value)
                config.logger.info(f"Logged parameter: {tag} = {value}")
            else:
                config.logger.warning(f"Parameter {tag} is None, skipping logging.")
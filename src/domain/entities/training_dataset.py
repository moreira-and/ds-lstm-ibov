# domain/entities/training_dataset.py
from domain.enums.problem_type import ProblemType
from typing import Any

class TrainingDataset:
    def __init__(
        self,
        data: Any,
        validation_size: int,
        sequence_length: int,
        problem_type: ProblemType,
        target_columns: list[str],
    ):
        self.data = data
        self.validation_size = validation_size
        self.sequence_length = sequence_length
        self.problem_type = problem_type
        self.target_columns = target_columns

        self._validate()

    def _validate(self):
        if self.validation_size < 0:
            raise ValueError("validation_size must be non-negative")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if not self.target_columns or not isinstance(self.target_columns, list):
            raise ValueError("You must specify at least one target column")

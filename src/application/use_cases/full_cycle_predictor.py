from src.domain.entities.data.raw_data import RawData
from src.domain.entities.data.predicted_data import PredictedData

from src.domain.interfaces.strategies.i_data_cleaner import IDataCleaner
from src.domain.interfaces.strategies.i_data_selector import IDataSelector
from src.domain.interfaces.strategies.i_data_adapter import IDataAdapter
from src.domain.interfaces.strategies.i_model import IModel

from src.domain.mocks.mock_data_cleaner import MockDataCleaner
from src.domain.mocks.mock_data_selector import MockDataSelector
from src.domain.mocks.mock_data_adapter import AdapterMetadata
from src.domain.mocks.mock_model import MockModel

class FullCyclePredictor:
    """
    Application use case for executing prediction using a trained model
    and an associated data adapter for transformation.
    """

    def __init__(
        self,        
        cleaner: IDataCleaner = None,
        selector: IDataSelector = None,
        adapter: IDataAdapter = None,
        model: IModel = None
    ) -> None:
        
        self.cleaner = cleaner or MockDataCleaner()
        self.selector = selector or MockDataSelector()
        self.adapter = adapter or AdapterMetadata()
        self.model = model or MockModel()

    def execute(self, data: RawData) -> PredictedData:
        """
        Transforms raw data, makes prediction, and reverses transformation.

        Args:
            data (RawData): Raw input data from upstream process.

        Returns:
            PredictedData: Final transformed prediction.
        """
        
        cleaned_data = self.cleaner.clean(data)
        selected_data = self.selector.select(cleaned_data)
        input_data = self.adapter.transform(selected_data)
        output_data = self.model.predict(input_data)
        predicted_data = self.adapter.inverse_transform(output_data)

        return predicted_data

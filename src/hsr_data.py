import pandas as pd
from sklearn.model_selection import train_test_split


class HsrData:
    def __init__(self, domain: str, label: str):
        self.DOMAIN = domain.capitalize()
        self.LABEL = label
        self.DATA_PATH = f"data/UNL_{self.DOMAIN}_measured_reflectance_param.csv"
        self._load_data()

    def _load_data(self) -> "HsrData":
        frame = pd.read_csv(self.DATA_PATH)
        self.hsr_columns = [str(wavelength) for wavelength in range(400, 1401)]
        self.feature_matrix = frame[self.hsr_columns]
        if self.LABEL in frame.columns:
            self.labels = frame[self.LABEL]
        else:
            raise NotImplementedError(f"{self.LABEL} not in the data frame.")

        return self

from abc import ABC, abstractmethod


class BasicReport(ABC):
    
    def __init__(self, title: str, **kwargs):
        self.title = title
        self.report = None

    @abstractmethod
    def generate_report(self) -> None:
        pass

    @abstractmethod
    def save_report(self, file_path: str, format: str = 'csv') -> None:
        """
        Save the generated report to a file.

        Parameters
        ----------
        file_path : str
            The path where the report will be saved.
        format : str, optional
            The format of the report file. Options are 'pdf', 'csv' (default is 'csv').
        """
        if self.report is None:
            raise ValueError("Report has not been generated yet.")
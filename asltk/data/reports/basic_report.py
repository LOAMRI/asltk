# from abc import ABC, abstractmethod


# class BasicReport(ABC):
#     """
#     This is an abstract base class for generating reports.
#     It provides a structure for creating reports with a title and methods
#     for generating and saving the report.

#     Args:
#         ABC: Abstract Base Class for defining abstract methods.
#     """

#     def __init__(self, title: str, **kwargs):
#         """
#         Initialize the BasicReport with a title.

#         Args:
#             title (str): The title of the report.
#         """
#         self.title = title
#         self.report = None

#     @abstractmethod
#     def generate_report(self) -> None:
#         """
#         Generate the report content.
#         This method should be implemented by subclasses to create the report content.
#         It should populate the `self.report` attribute with the report data.
#         The report can be in any format, such as text, HTML, or a structured data format.
#         The specific implementation will depend on the type of report being generated.
#         """
#         pass

#     @abstractmethod
#     def save_report(self, file_path: str, format: str = 'csv') -> None:
#         """
#         Save the generated report to a file.

#         Parameters
#         ----------
#         file_path : str
#             The path where the report will be saved.
#         format : str, optional
#             The format of the report file. Options are 'pdf', 'csv' (default is 'csv').
#         """
#         if self.report is None:
#             raise ValueError('Report has not been generated yet.')

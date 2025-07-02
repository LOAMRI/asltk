import pytest

from asltk.data.reports import BasicReport


def test_basic_report_create_object_success():
    """
    Test the BasicReport class.
    This test checks if the report can be generated and saved correctly.
    """
    # Create an instance of BasicReport
    class TestClass(BasicReport):
        def __init__(self, title='Test Report'):
            super().__init__(title=title)

        def generate_report(self):
            pass

        def save_report(self, path):
            pass

    report = TestClass()

    assert isinstance(report, BasicReport)
    assert report.title == 'Test Report'
    assert report.report is None


def test_basic_report_create_object_raise_error_when_report_not_generated_yet():
    """
    Test the BasicReport class.
    This test checks if the report can be generated and saved correctly.
    """
    # Create an instance of BasicReport
    class TestClass(BasicReport):
        def __init__(self, title='Test Report'):
            super().__init__(title=title)

        def generate_report(self):
            pass

        def save_report(self, path):
            # Call the parent method to get the validation check
            super().save_report(path)

    report = TestClass()
    with pytest.raises(Exception) as e:
        report.save_report('dummy_path')

    assert 'Report has not been generated yet' in str(e.value)


def test_basic_report_generate_report_abstract_method():
    """
    Test that the generate_report method raises NotImplementedError.
    This test checks if the abstract method is correctly defined and raises an error when called.
    """

    with pytest.raises(Exception) as e:
        report = BasicReport(title='Test Report')

    assert 'abstract class BasicReport without an implementation' in str(
        e.value
    )

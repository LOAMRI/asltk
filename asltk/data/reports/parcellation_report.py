from asltk.data.brain_atlas import BrainAtlas
from asltk.data.reports.basic_report import BasicReport


class ParcellationReport(BasicReport):
    def __init__(self, atlas_name: str = 'MNI2009'):
        pass

    def generate_report(self):
        # Report structure:
        # Description section:
        # - Report information: date
        # - Brain Atlas: Name and description
        # - Brain Regions: List of regions with their labels and descriptions
        # - Subject Information: Subject filename, image dimensions, image type, image resolution
        # Illustration section:
        # - Brain atlas illustration: Image of the brain atlas with regions labeled (5 slices I-S)
        # - Subject illustration: Image of subject's brain without parcellation (5 slices I-S)
        # - Subject illustration: Image of the subject's brain with parcellation overlay (5 slices I-S)
        # Parcellation section:
        # - Table with parcellation statistics:
        #   - Region label
        #   - Region name
        #   - Number of voxels
        #   - Volume in mm³
        #   - Average intensity
        #   - Std. deviation of intensity
        #   - Minimum intensity
        #   - Maximum intensity
        #   - Coefficient of variation (CV)
        pass

    def save_report(self, file_path: str, format: str = 'csv'):
        pass

# Brain atlas list for ASLtk
# All the data are storage in the Kaggle ASLtk project
# When a new data is called, then the brain atlas is allocated locally
import os

from asltk.data.kaggle_tools import download_brain_atlas

# TODO https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
# TODO MNI2009 - Check the FSL compatible atlas


class BrainAtlas:

    ATLAS_JSON_PATH = os.path.join(os.path.dirname(__file__))

    def __init__(self):
        pass

    def get_atlas_url(self, atlas_name: str):
        """
        Get the dataset URL of the atlas from the ASLtk database.
        The atlas URL is the base Kaggle URL where the atlas is stored.

        The `atlas_name` should be the name of the atlas as it is stored in the ASLtk database.
        To check all the available atlases, you can use the `list_atlas` method.

        Args:
            atlas_name (str): The name of the atlas to retrieve the URL for.

        Raises:
            ValueError: If the atlas name is not found in the database.

        Returns:
            str: The Kaggle dataset URL of the atlas if it exists, otherwise None.
        """
        if atlas_name not in self.list_atlas():
            raise ValueError(f'Atlas {atlas_name} not found in the database.')

        atlas_path = os.path.join(self.ATLAS_JSON_PATH, f'{atlas_name}.json')
        with open(atlas_path, 'r') as f:
            atlas_info = f.read()

        return atlas_info.get('dataset_url', None)

    def get_atlas_labels(self, atlas_name: str):
        pass

    def get_atlas_info(self, atlas_name: str):
        pass

    def list_atlas(self):
        """
        List all the available brain atlases in the ASLtk database.
        The atlas names are derived from the JSON files stored in the `ATLAS_JSON_PATH`.

        The JSON names should follow the format `<atlas_name>.json`.
        The atlas names are returned without the `.json` extension.

        Returns:
            list(str): List of atlas names available in the ASLtk database.
        """
        return [
            f[:-5]
            for f in os.listdir(self.ATLAS_JSON_PATH)
            if f.endswith('.json')
        ]

    def _check_atlas_name(self, atlas_name: str):
        # check if the atlas_name exist into the ASLtk atlas database
        pass

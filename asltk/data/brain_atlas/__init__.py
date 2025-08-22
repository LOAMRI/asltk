# Brain atlas list for ASLtk
# All the data are storage in the Kaggle ASLtk project
# When a new data is called, then the brain atlas is allocated locally
import json
import os
import time
from datetime import datetime

import kagglehub


class BrainAtlas:

    ATLAS_JSON_PATH = os.path.join(os.path.dirname(__file__))
    # Class-level variable to track API calls
    _last_api_call = None
    _min_call_interval = 2  # Minimum seconds between API calls

    def __init__(self, atlas_name: str = 'MNI2009', resolution: str = '1mm'):
        """
        Initializes the BrainAtlas class with a specified atlas name.
        If no atlas name is provided, it defaults to 'MNI2009'.

        Args:
            atlas_name (str, optional):  The name of the atlas to be used. Defaults to 'MNI2009'.
        """
        self._check_resolution_input(resolution)

        self._chosen_atlas = None
        self._resolution = resolution

        self.set_atlas(atlas_name)

    def set_atlas(self, atlas_name: str):
        """
        Sets the brain atlas to be used for ASLtk operations.
        This method checks if the provided atlas name exists in the available atlas database.
        If found, it loads the corresponding atlas JSON file, downloads the atlas data using the
        URL specified in the JSON (via kagglehub), and updates the atlas data with the local file
        location. The selected atlas data is then stored internally for further use.

        Notes:
        The atlas name should match one of the available atlases in the ASLtk database.
        To see all the available atlases, you can use the `list_atlas` method.

        Args:
            atlas_name (str): The name of the atlas to set. Must match an available atlas.

        Raises:
            ValueError: If the atlas name is not found in the database or if there is an error
                        downloading the atlas data.
        """
        if atlas_name not in self.list_atlas():
            raise ValueError(f'Atlas {atlas_name} not found in the database.')

        atlas_path = os.path.join(self.ATLAS_JSON_PATH, f'{atlas_name}.json')
        with open(atlas_path, 'r') as f:
            atlas_data = json.load(f)

        # Apply rate limiting before API call
        self._respect_rate_limits()

        # Add the current atlas file location in the atlas data
        try:
            path = kagglehub.dataset_download(
                atlas_data.get('dataset_url', None)
            )
            # Update the last API call timestamp after successful download
            BrainAtlas._last_api_call = datetime.now()
        except Exception as e:
            raise ValueError(f'Error downloading the atlas: {e}')

        # Assuming the atlas_data is a dictionary, we can add the path to it
        atlas_data['atlas_file_location'] = path
        # Assuming the atlas data contains a key for T1-weighted and Label image data
        atlas_data['resolution'] = self._resolution
        atlas_data['t1_data'] = os.path.join(path, self._collect_t1(path))
        atlas_data['label_data'] = os.path.join(
            path, self._collect_label(path)
        )

        self._chosen_atlas = atlas_data

    def get_atlas(self):
        """
        Get the current brain atlas data.

        Returns:
            dict: The current atlas data.
        """
        return self._chosen_atlas

    def set_resolution(self, resolution: str):
        self._check_resolution_input(resolution)
        self._resolution = resolution

    def get_resolution(self):
        return self._resolution

    def get_atlas_url(self, atlas_name: str):
        """
        Get the brain atlas URL of the chosen format in the ASLtk database.
        The atlas URL is the base Kaggle URL where the atlas is stored.


        Notes:
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

        try:
            atlas_url = self._chosen_atlas.get('dataset_url', None)
        except AttributeError:
            raise ValueError(
                f'Atlas {atlas_name} is not set or does not have a dataset URL.'
            )

        return atlas_url

    def get_atlas_labels(self):
        """
        Get the labels of the chosen brain atlas.
        This method retrieves the labels associated with the current atlas.
        Notes:
        The labels are typically used for parcellation or segmentation tasks in brain imaging.

        Returns:
            dict: The labels of the current atlas if available, otherwise None.
        """
        return self._chosen_atlas.get('labels', None)

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

    def _collect_t1(self, path: str):  # pragma: no cover
        """
        Collect the T1-weighted image data from the atlas directory.
        Args:
            path (str): The path to the atlas directory.
        Returns:
            str: The filename of the T1-weighted image data.
        """
        t1_file = next(
            (f for f in os.listdir(path) if self._resolution + '_t1' in f),
            None,
        )
        if t1_file is None:
            raise ValueError(
                f"No file with '_t1_' and resolution {self._resolution} found in the atlas directory: {path}"
            )

        return t1_file

    def _collect_label(self, path: str):   # pragma: no cover
        """
        Collect the label file from the atlas directory.
        Args:
            path (str): The path to the atlas directory.
        Returns:
            str: The filename of the label file.
        """
        label_file = next(
            (f for f in os.listdir(path) if self._resolution + '_label' in f),
            None,
        )
        if label_file is None:
            raise ValueError(
                f"No file with '_label' and resolution {self._resolution} found in the atlas directory: {path}"
            )

        return label_file

    def _check_resolution_input(self, resolution):
        valid_resolutions = ['1mm', '2mm']
        if resolution not in valid_resolutions:
            raise ValueError(
                f"Invalid resolution '{resolution}'. Valid options are: {valid_resolutions}"
            )

    @classmethod
    def _respect_rate_limits(cls):
        """
        Ensures API calls respect rate limits by adding delay if necessary.
        This helps prevent 429 Too Many Requests errors.

        The method enforces a minimum interval between consecutive API calls
        by sleeping if the last call was too recent.
        """
        if cls._last_api_call is not None:
            elapsed = (datetime.now() - cls._last_api_call).total_seconds()
            if elapsed < cls._min_call_interval:
                time.sleep(cls._min_call_interval - elapsed)

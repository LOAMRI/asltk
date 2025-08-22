import pytest

from asltk.data.brain_atlas import BrainAtlas


def test_set_atlas_raise_error_when_atlas_name_does_not_exist():
    """
    Test if setting an atlas raises an error when the atlas name does not exist.
    """
    atlas = BrainAtlas()
    with pytest.raises(ValueError) as e:
        atlas.set_atlas('non_existent_atlas')

    assert 'not found in the database' in str(e.value)


def test_list_all_atlas():
    """
    Test if the BrainAtlas class can list all available atlases.
    """
    atlas = BrainAtlas()
    atlases = atlas.list_atlas()
    assert isinstance(atlases, list), 'The list of atlases should be a list.'
    assert len(atlases) > 0, 'There should be at least one atlas available.'


def test_get_atlas_url():
    """
    Test if the BrainAtlas class can retrieve the URL of a known atlas.
    """
    atlas = BrainAtlas(atlas_name='MNI2009')
    url = atlas.get_atlas_url('MNI2009')
    assert isinstance(url, str)   # The URL should be a string.
    assert 'loamri' in url


def test_get_atlas_labels():
    """
    Test if the BrainAtlas class can retrieve labels for a known atlas.
    """
    atlas = BrainAtlas(atlas_name='MNI2009')
    labels = atlas.get_atlas_labels()
    assert isinstance(labels, dict)   # 'Labels should be a dictionary.'
    assert (
        len(labels) > 0
    )   # 'There should be at least one label in the atlas.'


@pytest.mark.parametrize('known_atlas', ['AAL', 'HOCSA2006', 'AAT'])
def test_list_all_atlas_contains_known_atlas_parametrized(known_atlas):
    """
    Test if known atlases are present in the list of atlases.
    """
    atlas = BrainAtlas()
    atlases = atlas.list_atlas()
    assert any(
        known_atlas.lower() in a.lower() for a in atlases
    ), f"Known atlas '{known_atlas}' should be in the list."


def test_list_all_atlas_contains_known_atlas():
    """
    Test if a known atlas is present in the list of atlases.
    """
    atlas = BrainAtlas()
    atlases = atlas.list_atlas()
    # Replace 'AAL' with a known atlas name if different
    assert any(
        'aal' in a.lower() for a in atlases
    ), "Known atlas 'AAL' should be in the list."


def test_list_all_atlas_unique_names():
    """
    Test that the list of atlases does not contain duplicates.
    """
    atlas = BrainAtlas()
    atlases = atlas.list_atlas()
    assert len(atlases) == len(set(atlases)), 'Atlas names should be unique.'


def test_list_all_atlas_string_type():
    """
    Test that all atlas names are strings.
    """
    atlas = BrainAtlas()
    atlases = atlas.list_atlas()
    assert all(
        isinstance(a, str) for a in atlases
    ), 'All atlas names should be strings.'


def test_get_atlas_url_raise_error_when_atlas_name_does_not_exist():
    atlas = BrainAtlas()
    with pytest.raises(ValueError) as e:
        atlas.get_atlas_url('non_existent_atlas')

    assert 'not found in the database' in str(e)


@pytest.mark.parametrize(
    'atlas_name',
    [
        'MNI2009',
        'AAL32024',
        'HOCSA2006',
        'AAT2022',
        'AICHA2021',
        'DKA2006',
        'FCA7N2011',
        'HA2003',
        'JHA2005',
        'LGPHCC2022',
        'AAT2022',
    ],
)
def test_brain_atlas_creation_with_various_names(atlas_name):
    """
    Test creating BrainAtlas objects with different valid atlas names.
    """
    try:
        atlas = BrainAtlas(atlas_name=atlas_name)
        assert isinstance(atlas.get_atlas(), dict)
    except ValueError as e:
        if '429 Client Error: Too Many Requests' in str(e):
            pytest.skip(
                f'Skipping test for {atlas_name} due to Kaggle API rate limit: {e}'
            )
        else:
            raise


@pytest.mark.parametrize(
    'atlas_name',
    [
        'MNI2009',
        'AAL32024',
        'HOCSA2006',
        'AAT2022',
        'AICHA2021',
        'DKA2006',
        'FCA7N2011',
        'HA2003',
        'JHA2005',
        'LGPHCC2022',
        'AAT2022',
    ],
)
def test_brain_atlas_creation_with_various_names_2mm_resolution(atlas_name):
    """
    Test creating BrainAtlas objects with different valid atlas names and 2mm resolution.
    """
    try:
        atlas = BrainAtlas(atlas_name=atlas_name, resolution='2mm')
        assert isinstance(atlas.get_atlas(), dict)
    except ValueError as e:
        if '429 Client Error: Too Many Requests' in str(e):
            pytest.skip(
                f'Skipping test for {atlas_name} due to Kaggle API rate limit: {e}'
            )
        else:
            raise


@pytest.mark.parametrize(
    'wrong_resolution',
    [
        ('1'),
        ('2'),
        ('3mm'),
        ('1.5mm'),
        ('4mm'),
        ('1x1x1'),
        ('2x2x2'),
        (1),
        (2),
    ],
)
def test_brain_atlas_constructor_raise_error_wrong_resolution(
    wrong_resolution,
):
    """
    Test that the BrainAtlas constructor raises an error for invalid resolution.
    """
    with pytest.raises(ValueError) as e:
        BrainAtlas(resolution=wrong_resolution)

    assert 'Invalid resolution' in str(e.value)


def test_atlas_download_failure(mocker):
    """
    Test that appropriate error is raised when atlas download fails.
    """
    atlas = BrainAtlas()
    # Mock the kagglehub.dataset_download function to raise an exception
    mock_download = mocker.patch(
        'kagglehub.dataset_download', side_effect=Exception('Connection error')
    )

    # Attempt to set an atlas that would trigger the download
    with pytest.raises(ValueError) as e:
        atlas.set_atlas('MNI2009')  # This should try to download the atlas

    # Verify the error message contains the expected text
    assert 'Error downloading the atlas' in str(e.value)
    assert 'Connection error' in str(e.value)

    # Verify that the mocked function was called
    mock_download.assert_called_once()


def test_atlas_url_raises_error_when_atlas_not_set():
    """
    Test that appropriate error is raised when trying to get atlas URL
    without setting an atlas first.
    """
    atlas = BrainAtlas()
    atlas._chosen_atlas = None  # Simulate that no atlas is set
    # Don't set any atlas, which should cause an AttributeError in the implementation
    # that's caught and converted to a ValueError
    with pytest.raises(Exception) as e:
        # Access the private method directly since we want to test the specific exception handling
        atlas.get_atlas_url('MNI2009')

    # Verify the error message
    assert 'is not set or does not have a dataset URL' in str(e.value)


def test_brain_atlas_get_resolution():
    """
    Test the get_resolution method of the BrainAtlas class.
    """
    atlas = BrainAtlas()
    atlas.set_resolution('2mm')
    assert atlas.get_resolution() == '2mm'


def test_brain_atlas_set_resolution():
    """
    Test the set_resolution method of the BrainAtlas class.
    """
    atlas = BrainAtlas()
    atlas.set_resolution('2mm')
    assert atlas.get_resolution() == '2mm'

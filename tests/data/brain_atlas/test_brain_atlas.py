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
    ],
)
def test_brain_atlas_creation_with_various_names(atlas_name):
    """
    Test creating BrainAtlas objects with different valid atlas names.
    """
    atlas = BrainAtlas(atlas_name=atlas_name)
    assert isinstance(atlas.get_atlas(), dict)

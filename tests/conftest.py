from pathlib import Path

import kagglehub
import pytest
import SimpleITK as sitk

from asltk.data.brain_atlas import BrainAtlas


def _resample_image(source_path: Path, target_path: Path, target_size):
    source_image = sitk.ReadImage(str(source_path))
    source_size = source_image.GetSize()
    source_spacing = source_image.GetSpacing()

    target_spacing = [
        source_spacing[index] * source_size[index] / target_size[index]
        for index in range(3)
    ]

    resampled_image = sitk.Resample(
        source_image,
        target_size,
        sitk.Transform(),
        sitk.sitkLinear,
        source_image.GetOrigin(),
        target_spacing,
        source_image.GetDirection(),
        0,
        source_image.GetPixelID(),
    )

    sitk.WriteImage(resampled_image, str(target_path))


@pytest.fixture(scope='session')
def kaggle_brain_atlas_fixture_dir(tmp_path_factory):
    fixture_dir = tmp_path_factory.mktemp('brain_atlas_fixture')
    source_image = Path(__file__).resolve().parent / 'files' / 't1-mri.nrrd'

    target_size = [182, 218, 182]
    for resolution in ('1mm', '2mm'):
        for kind in ('t1', 'label'):
            target_path = fixture_dir / f'fixture_{resolution}_{kind}.nrrd'
            _resample_image(source_image, target_path, target_size)

    return fixture_dir


@pytest.fixture(autouse=True)
def disable_kaggle_download_for_unit_tests(
    monkeypatch, request, kaggle_brain_atlas_fixture_dir
):
    if request.node.get_closest_marker('integration') is not None:
        return

    monkeypatch.setattr(
        kagglehub,
        'dataset_download',
        lambda *args, **kwargs: str(kaggle_brain_atlas_fixture_dir),
    )
    monkeypatch.setattr(
        BrainAtlas,
        '_respect_rate_limits',
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(BrainAtlas, '_last_api_call', None, raising=False)

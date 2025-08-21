import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.data.brain_atlas import BrainAtlas
from asltk.registration.asl_normalization import asl_template_registration
from asltk.utils.io import ImageIO

SEP = os.sep
M0_ORIG = (
    f'tests' + SEP + 'files' + SEP + 'registration' + SEP + 'm0_mean.nii.gz'
)
M0_RIGID = (
    f'tests'
    + SEP
    + 'files'
    + SEP
    + 'registration'
    + SEP
    + 'm0_mean-rigid-25degrees.nrrd'
)
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'


def test_asl_template_registration_success():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
    # Reducing pcasl size to not exceed memory limits
    short_data = ImageIO(
        image_array=pcasl_orig('pcasl').get_as_numpy()[:3, :, :, :]
    )
    pcasl_orig.set_image(short_data, 'pcasl')
    # pcasl_orig = ASLData(
    #     pcasl='/home/antonio/Imagens/loamri-samples/20240909/pcasl.nii.gz',
    #     m0='/home/antonio/Imagens/loamri-samples/20240909/m0.nii.gz',
    #     average_m0=True,
    # )
    # asl_data_mask = np.ones_like(pcasl_orig('m0'), dtype=bool)

    (
        asl_data_registered,
        trans_mtxs,
        additional_maps_normalized,
    ) = asl_template_registration(
        pcasl_orig,
        atlas_reference='MNI2009',
        verbose=True,
    )

    assert isinstance(asl_data_registered, ASLData)
    assert isinstance(trans_mtxs, list)
    assert isinstance(additional_maps_normalized, list)


def test_asl_template_registration_invalid_input_type():
    with pytest.raises(TypeError) as e:
        asl_template_registration('not_asldata')
    assert str(e.value) == 'Input must be an ASLData object.'


def test_asl_template_registration_raise_error_if_m0_volume_not_present_at_input_asl_data():
    pcasl_orig = ASLData(pcasl=PCASL_MTE)
    with pytest.raises(ValueError) as e:
        asl_template_registration(pcasl_orig)
    assert 'M0 image is required for normalization' in str(e.value)


@pytest.mark.parametrize(
    'atlas_reference',
    [('invalid_atlas'), ('/tmp/invalid_path.nii.gz')],
)
def test_asl_template_registration_checks_atlas_reference_types(
    atlas_reference,
):
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
    with pytest.raises(Exception) as e:
        asl_template_registration(pcasl_orig, atlas_reference=atlas_reference)
    assert isinstance(str(e.value), str)

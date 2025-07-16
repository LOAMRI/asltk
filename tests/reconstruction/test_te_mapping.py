import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction.t2_mapping import T2Scalar_ASLMapping
from asltk.utils import load_image

SEP = os.sep

T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
PCASL_MDW = f'tests' + SEP + 'files' + SEP + 'pcasl_mdw.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'

asldata_te = ASLData(
    pcasl=PCASL_MTE,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    te_values=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
)


def test_t2_scalar_asl_mapping_initialization():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)

    assert isinstance(t2_mapping, T2Scalar_ASLMapping)
    assert isinstance(t2_mapping._asl_data, ASLData)
    assert isinstance(t2_mapping._brain_mask, np.ndarray)
    assert t2_mapping._t2_maps is None
    assert t2_mapping._mean_t2s is None


def test_t2_scalar_mapping_raise_error_if_asl_data_do_not_has_te_values():
    asldata = ASLData(pcasl=PCASL_MTE, m0=M0)
    with pytest.raises(ValueError) as error:
        T2Scalar_ASLMapping(asldata)
    assert str(error.value) == 'ASLData must provide TE and PLD values.'


def test_t2_scalar_mapping_raise_error_if_asl_data_do_not_has_pld_values():
    asldata = ASLData(pcasl=PCASL_MTE, m0=M0, te_values=asldata_te.get_te())
    with pytest.raises(ValueError) as error:
        T2Scalar_ASLMapping(asldata)
    assert str(error.value) == 'ASLData must provide TE and PLD values.'


def test_t2_scalar_mapping_success_construction_t2_map():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)

    out = t2_mapping.create_map()

    assert isinstance(out['t2'], np.ndarray)
    assert out['t2'].ndim == 4  # Expecting a 4D array
    assert out['mean_t2'] is not None
    assert len(out['mean_t2']) == len(
        asldata_te.get_pld()
    )  # One mean T2 per PLD


# TODO Test for asl data that has more than PLD and TEs (for instance an asldata with dw included as well)

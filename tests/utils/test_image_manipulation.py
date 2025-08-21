import os
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

from asltk import asldata
from asltk.models import signal_dynamic
from asltk.utils.image_manipulation import (
    collect_data_volumes,
    select_reference_volume,
)
from asltk.utils.io import ImageIO

SEP = os.sep
T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'


def test_asl_model_buxton_return_sucess_list_of_values():
    buxton_values = signal_dynamic.asl_model_buxton(
        tau=[1, 2, 3], w=[10, 20, 30], m0=1000, cbf=450, att=1500
    )
    assert len(buxton_values.tolist()) == 3
    assert type(buxton_values) == np.ndarray


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 2]), ([100.1, 200.0, 'text'])]
)
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = signal_dynamic.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'tau list must contain float or int values'


@pytest.mark.parametrize('input', [('a'), (2), (100.1)])
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs_type(input):
    with pytest.raises(Exception) as e:
        buxton_values = signal_dynamic.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert (
        e.value.args[0] == 'tau parameter must be a list or tuple of values.'
    )


@pytest.mark.parametrize('input', [(['a']), (['2']), (['100.1'])])
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs_values(input):
    with pytest.raises(Exception) as e:
        buxton_values = signal_dynamic.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'tau list must contain float or int values'


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 2]), ([100.1, 200.0, np.ndarray])]
)
def test_asl_model_buxton_w_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = signal_dynamic.asl_model_buxton(
            tau=[10, 20, 30], w=input, m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'w list must contain float or int values'


@pytest.mark.parametrize('input', [('a'), (1), (100.1), (np.ndarray)])
def test_asl_model_buxton_w_raise_errors_with_wrong_inputs_not_list(input):
    with pytest.raises(Exception) as e:
        buxton_values = signal_dynamic.asl_model_buxton(
            tau=[10, 20, 30], w=input, m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'w parameter must be a list or tuple of values.'


def test_asl_model_buxton_runs_with_inner_if_clauses():
    buxton_values = signal_dynamic.asl_model_buxton(
        tau=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
        w=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        m0=3761480.0,
        cbf=0.00001,
        att=1500,
    )
    assert len(buxton_values.tolist()) == 7
    assert type(buxton_values) == np.ndarray


def test_asl_model_multi_te_return_sucess_list_of_values():
    multite_values = signal_dynamic.asl_model_multi_te(
        tau=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
        w=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        te=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
        m0=3761480.0,
        cbf=0.00001,
        att=1500,
    )
    assert len(multite_values) == 7
    assert type(multite_values) == np.ndarray


def test_collect_data_volumes_return_correct_list_of_volumes_4D_data():
    data = np.ones((2, 30, 40, 15))
    data[0, :, :, :] = data[0, :, :, :] * 10
    data[1, :, :, :] = data[1, :, :, :] * 20
    image = ImageIO(image_array=data)
    collected_volumes, _ = collect_data_volumes(image)
    assert len(collected_volumes) == 2
    assert collected_volumes[0].get_as_numpy().shape == (30, 40, 15)
    assert np.mean(collected_volumes[0].get_as_numpy()) == 10
    assert np.mean(collected_volumes[1].get_as_numpy()) == 20


def test_collect_data_volumes_return_correct_list_of_volumes_5D_data():
    data = np.ones((2, 2, 30, 40, 15))
    data[0, 0, :, :, :] = data[0, 0, :, :, :] * 10
    data[0, 1, :, :, :] = data[0, 1, :, :, :] * 10
    data[1, 0, :, :, :] = data[1, 0, :, :, :] * 20
    data[1, 1, :, :, :] = data[1, 1, :, :, :] * 20
    data = ImageIO(image_array=data)
    collected_volumes, _ = collect_data_volumes(data)
    assert len(collected_volumes) == 4
    assert collected_volumes[0].get_as_numpy().shape == (30, 40, 15)
    assert np.mean(collected_volumes[0].get_as_numpy()) == 10
    assert np.mean(collected_volumes[1].get_as_numpy()) == 10
    assert np.mean(collected_volumes[2].get_as_numpy()) == 20
    assert np.mean(collected_volumes[3].get_as_numpy()) == 20


def test_collect_data_volumes_error_if_input_is_not_numpy_array():
    data = [1, 2, 3]
    with pytest.raises(Exception) as e:
        collected_volumes, _ = collect_data_volumes(data)
    assert 'data is not an ImageIO object' in e.value.args[0]


def test_collect_data_volumes_error_if_input_is_less_than_3D():
    data = ImageIO(image_array=np.ones((30, 40)))
    with pytest.raises(Exception) as e:
        collected_volumes, _ = collect_data_volumes(data)
    assert 'data is not a 3D volume or higher dimensions' in e.value.args[0]


@pytest.mark.parametrize('method', ['snr', 'mean'])
def test_select_reference_volume_returns_correct_volume_and_index_with_sample_images(
    method,
):
    asl = asldata.ASLData(pcasl=PCASL_MTE, m0=M0)

    ref_volume, idx = select_reference_volume(asl, method=method)

    assert (
        ref_volume.get_as_numpy().shape
        == asl('pcasl').get_as_numpy()[0][0].shape
    )
    assert idx != 0


@pytest.mark.parametrize(
    'method', [('invalid_method'), (123), (['mean']), ({'method': 'snr'})]
)
def test_select_reference_volume_raise_error_invalid_method(method):
    asl = asldata.ASLData(pcasl=PCASL_MTE, m0=M0)

    with pytest.raises(Exception) as e:
        select_reference_volume(asl, method=method)
    assert 'Invalid method' in e.value.args[0]


def test_select_reference_volume_raise_error_wrong_roi():
    asl = asldata.ASLData(pcasl=PCASL_MTE, m0=M0)

    with pytest.raises(Exception) as e:
        select_reference_volume(asl, roi='invalid_roi')
    assert 'ROI must be an ImageIO object' in e.value.args[0]


def test_select_reference_volume_raise_error_wrong_4D_roi():
    asl = asldata.ASLData(pcasl=PCASL_MTE, m0=M0)
    roi = ImageIO(
        image_array=np.array(
            [asl('m0').get_as_numpy(), asl('m0').get_as_numpy()]
        )
    )

    with pytest.raises(Exception) as e:
        select_reference_volume(asl, roi=roi)
    assert 'ROI must be a 3D array' in e.value.args[0]


def test_select_reference_volume_raise_error_wrong_list_image_input_images():
    wrong_input_list = ['wrong_input1', 'wrong_input2']

    with pytest.raises(Exception) as e:
        select_reference_volume(wrong_input_list)
    assert 'asl_data must be an ASLData object' in e.value.args[0]


def test_select_reference_volume_raise_error_wrong_method():
    asl = asldata.ASLData(pcasl=PCASL_MTE, m0=M0)

    with pytest.raises(Exception) as e:
        select_reference_volume(asl, method='invalid_method')
    assert 'Invalid method' in e.value.args[0]

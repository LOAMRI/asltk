import os

import numpy as np
import pytest
import SimpleITK as sitk
import tempfile

from asltk import asldata, utils

SEP = os.sep
T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'


def test_load_image_pcasl_type_update_object_image_reference():
    img = utils.load_image(PCASL_MTE)
    assert isinstance(img, np.ndarray)


def test_load_image_m0_type_update_object_image_reference():
    img = utils.load_image(M0)
    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize(
    'input',
    [
        ('/wrong/path'),
        ('not-a-path'),
        (f'tests' + SEP + 'files' + SEP + 'no-image.nrrd'),
    ],
)
def test_load_image_attest_fullpath_is_valid(input):
    with pytest.raises(Exception) as e:
        utils.load_image(input)
    assert 'does not exist.' in e.value.args[0]


@pytest.mark.parametrize(
    'input', [('out.nrrd'), ('out.nii'), ('out.mha'), ('out.tif')]
)
def test_save_image_success(input, tmp_path):
    img = utils.load_image(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    utils.save_image(img, full_path)
    assert os.path.exists(full_path)
    read_file = sitk.ReadImage(full_path)
    assert read_file.GetSize() == sitk.ReadImage(T1_MRI).GetSize()


@pytest.mark.parametrize(
    'input', [('out.nrr'), ('out.n'), ('out.m'), ('out.zip')]
)
def test_save_image_throw_error_invalid_formatt(input, tmp_path):
    img = utils.load_image(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    with pytest.raises(Exception) as e:
        utils.save_image(img, full_path)


def test_asl_model_buxton_return_sucess_list_of_values():
    buxton_values = utils.asl_model_buxton(
        tau=[1, 2, 3], w=[10, 20, 30], m0=1000, cbf=450, att=1500
    )
    assert len(buxton_values.tolist()) == 3
    assert type(buxton_values) == np.ndarray


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 2]), ([100.1, 200.0, 'text'])]
)
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'tau list must contain float or int values'


@pytest.mark.parametrize('input', [('a'), (2), (100.1)])
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs_type(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert (
        e.value.args[0] == 'tau parameter must be a list or tuple of values.'
    )


@pytest.mark.parametrize('input', [(['a']), (['2']), (['100.1'])])
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs_values(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'tau list must contain float or int values'


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 2]), ([100.1, 200.0, np.ndarray])]
)
def test_asl_model_buxton_w_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=[10, 20, 30], w=input, m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'w list must contain float or int values'


@pytest.mark.parametrize('input', [('a'), (1), (100.1), (np.ndarray)])
def test_asl_model_buxton_w_raise_errors_with_wrong_inputs_not_list(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=[10, 20, 30], w=input, m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'w parameter must be a list or tuple of values.'


def test_asl_model_buxton_runs_with_inner_if_clauses():
    buxton_values = utils.asl_model_buxton(
        tau=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
        w=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        m0=3761480.0,
        cbf=0.00001,
        att=1500,
    )
    assert len(buxton_values.tolist()) == 7
    assert type(buxton_values) == np.ndarray


def test_asl_model_multi_te_return_sucess_list_of_values():
    multite_values = utils.asl_model_multi_te(
        tau=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
        w=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        te=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
        m0=3761480.0,
        cbf=0.00001,
        att=1500,
    )
    assert len(multite_values) == 7
    assert type(multite_values) == np.ndarray


@pytest.mark.parametrize(
    'input_data,filename',
    [
        (PCASL_MTE, 'multi_te_asldata.pkl'),
    ],
)
def test_save_asl_data_data_sucess(input_data, filename, tmp_path):
    obj = asldata.ASLData(pcasl=input_data)
    out_file = tmp_path.as_posix() + os.sep + filename
    utils.save_asl_data(obj, out_file)
    assert os.path.exists(out_file)


@pytest.mark.parametrize(
    'input_data,filename',
    [
        (PCASL_MTE, 'multi_te_asldata.txt'),
        (PCASL_MTE, 'multi_te_asldata.nii'),
        (PCASL_MTE, 'multi_te_asldata.mha'),
        (PCASL_MTE, 'multi_te_asldata.nrrd'),
    ],
)
def test_save_asl_data_raise_error_filename_not_pkl(
    input_data, filename, tmp_path
):
    obj = asldata.ASLData(pcasl=PCASL_MTE)
    out_file = tmp_path.as_posix() + os.sep + filename
    with pytest.raises(Exception) as e:
        utils.save_asl_data(obj, out_file)
    assert e.value.args[0] == 'Filename must be a pickle file (.pkl)'


@pytest.mark.parametrize(
    'input_data,filename',
    [
        (PCASL_MTE, 'multi_te_asldata.pkl'),
    ],
)
def test_load_asl_data_sucess(input_data, filename, tmp_path):
    obj = asldata.ASLData(pcasl=input_data)
    out_file = tmp_path.as_posix() + os.sep + filename
    utils.save_asl_data(obj, out_file)
    loaded_obj = utils.load_asl_data(out_file)
    assert isinstance(loaded_obj, asldata.ASLData)
    assert loaded_obj('pcasl').shape == obj('pcasl').shape


@pytest.mark.parametrize(
    'input_bids,sub,sess,mod,suff',
    [
        ('./tests/files/bids-example/asl001', None, None, None, None),
        ('./tests/files/bids-example/asl001', 103, None, None, None),
        ('./tests/files/bids-example/asl001', None, None, 'asl', None),
        ('./tests/files/bids-example/asl001', 103, None, 'asl', None),
    ],
)
def test_load_image_using_BIDS_input_sucess(input_bids, sub, sess, mod, suff):
    loaded_obj = utils.load_image(
        full_path=input_bids,
        subject=sub,
        session=sess,
        modality=mod,
        suffix=suff,
    )
    assert isinstance(loaded_obj, np.ndarray)


@pytest.mark.parametrize(
    'input_data',
    [(tempfile.gettempdir())],
)
def test_load_image_using_not_valid_BIDS_input_raise_error(input_data):
    with pytest.raises(Exception) as e:
        loaded_obj = utils.load_image(input_data)
    assert 'is missing' in e.value.args[0]


@pytest.mark.parametrize(
    'input_bids,sub,sess,mod,suff',
    [
        ('./tests/files/bids-example/asl001', 2, None, None, None),
        ('./tests/files/bids-example/asl001', 502, None, 'flair', None),
        ('./tests/files/bids-example/asl001', 2, None, 'bold', None),
    ],
)
def test_load_image_raise_FileNotFoundError_not_matching_image_file(
    input_bids, sub, sess, mod, suff
):
    with pytest.raises(Exception) as e:
        loaded_obj = utils.load_image(
            full_path=input_bids,
            subject=sub,
            session=sess,
            modality=mod,
            suffix=suff,
        )
    assert 'ASL image file is missing' in e.value.args[0]

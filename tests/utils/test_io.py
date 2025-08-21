import os
import tempfile

import ants
import numpy as np
import pytest
import SimpleITK as sitk

from asltk import asldata
from asltk.models import signal_dynamic
from asltk.utils.io import (
    ImageIO,
    check_image_properties,
    check_path,
    clone_image,
    load_asl_data,
    save_asl_data,
)

SEP = os.sep
T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'


def test_load_image_pcasl_type_update_object_image_reference():
    img = ImageIO(PCASL_MTE)
    assert isinstance(img, ImageIO)


def test_load_image_m0_type_update_object_image_reference():
    img = ImageIO(M0)
    assert isinstance(img, ImageIO)


def test_load_image_m0_with_average_m0_option(tmp_path):
    img_4d = np.array(
        [ImageIO(M0).get_as_numpy(), ImageIO(M0).get_as_numpy()],
        dtype=np.float32,
    )
    multi_M0 = ImageIO(image_array=img_4d)
    tmp_file = tmp_path / 'temp_m0.nii.gz'
    multi_M0.save_image(str(tmp_file))
    img = ImageIO(str(tmp_file), average_m0=True)

    assert isinstance(img, ImageIO)
    assert len(img.get_as_numpy().shape) == 3


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
        ImageIO(input)
    assert 'does not exist.' in e.value.args[0]


@pytest.mark.parametrize(
    'input', [('out.nrrd'), ('out.nii'), ('out.mha'), ('out.tif')]
)
def test_save_image_success(input, tmp_path):
    img = ImageIO(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    img.save_image(full_path)
    assert os.path.exists(full_path)
    read_file = sitk.ReadImage(full_path)
    assert read_file.GetSize() == sitk.ReadImage(T1_MRI).GetSize()


@pytest.mark.parametrize(
    'input', [('out.nrr'), ('out.n'), ('out.m'), ('out.zip')]
)
def test_save_image_throw_error_invalid_formatt(input, tmp_path):
    img = ImageIO(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    with pytest.raises(Exception) as e:
        img.save_image(full_path)


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


@pytest.mark.parametrize(
    'input_data,filename',
    [
        (PCASL_MTE, 'multi_te_asldata.pkl'),
    ],
)
def test_save_asl_data_data_sucess(input_data, filename, tmp_path):
    obj = asldata.ASLData(pcasl=input_data)
    out_file = tmp_path.as_posix() + os.sep + filename
    save_asl_data(obj, out_file)
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
        save_asl_data(obj, out_file)
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
    save_asl_data(obj, out_file)
    loaded_obj = load_asl_data(out_file)
    assert isinstance(loaded_obj, asldata.ASLData)
    assert (
        loaded_obj('pcasl').get_as_numpy().shape
        == obj('pcasl').get_as_numpy().shape
    )


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
    loaded_obj = ImageIO(
        image_path=input_bids,
        subject=sub,
        session=sess,
        modality=mod,
        suffix=suff,
    )
    assert isinstance(loaded_obj, ImageIO)


@pytest.mark.parametrize(
    'input_data',
    [(tempfile.gettempdir())],
)
def test_load_image_using_not_valid_BIDS_input_raise_error(input_data):
    with pytest.raises(Exception) as e:
        loaded_obj = ImageIO(input_data)
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
        loaded_obj = ImageIO(
            image_path=input_bids,
            subject=sub,
            session=sess,
            modality=mod,
            suffix=suff,
        )
    assert 'ASL image file is missing' in e.value.args[0]


def test_load_image_from_bids_structure_returns_valid_array():
    bids_root = 'tests/files/bids-example/asl001'
    subject = 'Sub103'
    session = None
    modality = 'asl'
    suffix = None  # m0 is deleted, because it does not exist

    img = ImageIO(
        image_path=bids_root,
        subject=subject,
        session=session,
        modality=modality,
        suffix=suffix,
    )

    assert img is not None


@pytest.mark.parametrize(
    'input_data, type',
    [
        (np.random.rand(10, 10, 10), 'array'),
        (np.random.rand(10, 10, 10, 5), 'array'),
        (np.random.rand(5, 2, 10, 10, 10), 'array'),
        (T1_MRI, 'path'),
        (PCASL_MTE, 'path'),
        (M0, 'path'),
    ],
)
def test_ImageIO_constructor_success_with_image_array(input_data, type):
    """Test ImageIO constructor with an image array."""
    if type == 'array':
        img_array = input_data
        io = ImageIO(image_array=img_array)
        assert isinstance(io, ImageIO)
        assert np.array_equal(io.get_as_numpy(), img_array)
    elif type == 'path':
        img_path = input_data
        io = ImageIO(image_path=img_path)
        assert isinstance(io, ImageIO)
        assert io.get_as_numpy() is not None


def test_ImageIO_str_representation():
    """Test the __str__ method of ImageIO."""
    img = ImageIO(T1_MRI)
    representation = str(img)
    assert 'Path: ' in representation
    assert 'Dimension: 3' in representation
    assert (
        'Spacing: (15.000015000015, 15.000015000015, 14.884615384615385)'
        in representation
    )
    assert 'average_m0: False' in representation
    assert 'verbose: False' in representation
    assert 'Subject: None' in representation
    assert 'Session: None' in representation
    assert 'Modality: None' in representation
    assert 'Suffix: None' in representation


def test_ImageIO_set_image_path_sucess():
    """Test setting a new image path."""
    img = ImageIO(T1_MRI)
    new_path = PCASL_MTE
    img.set_image_path(new_path)
    assert img.get_image_path() == new_path
    assert img.get_as_numpy() is not None


def test_ImageIO_set_image_path_invalid_path():
    """Test setting an invalid image path."""
    img = ImageIO(T1_MRI)
    invalid_path = 'invalid/path/to/image.nii'
    with pytest.raises(Exception) as e:
        img.set_image_path(invalid_path)
    assert 'does not exist.' in e.value.args[0]


def test_ImageIO_get_image_path():
    """Test getting the image path."""
    img = ImageIO(T1_MRI)
    assert img.get_image_path() == T1_MRI


def test_ImageIO_get_as_sitk_sucess():
    """Test getting the image as a SimpleITK object."""
    img = ImageIO(T1_MRI)
    sitk_img = img.get_as_sitk()
    assert isinstance(sitk_img, sitk.Image)
    assert sitk_img.GetSize() == sitk.ReadImage(T1_MRI).GetSize()


def test_ImageIO_get_as_sitk_raise_error_no_image_loaded():
    """Test getting the image as SimpleITK when no image is loaded."""
    img = ImageIO(image_array=np.ones((5, 5, 5)))
    img._image_as_sitk = None  # Force no image loaded
    with pytest.raises(Exception) as e:
        img.get_as_sitk()
    assert (
        e.value.args[0]
        == 'Image is not loaded as SimpleITK. Please load the image first.'
    )


def test_ImageIO_get_as_ants_sucess():
    """Test getting the image as an ANTs object."""
    img = ImageIO(T1_MRI)
    ants_img = img.get_as_ants()
    assert ants_img is not None
    assert ants_img.dimension == 3
    assert isinstance(ants_img, ants.ANTsImage)


def test_ImageIO_get_as_ants_raise_error_no_image_loaded():
    """Test getting the image as ANTs when no image is loaded."""
    img = ImageIO(image_array=np.ones((5, 5, 5)))
    img._image_as_ants = None  # Force no image loaded
    with pytest.raises(Exception) as e:
        img.get_as_ants()
    assert (
        e.value.args[0]
        == 'Image is not loaded as ANTsPy. Please load the image first.'
    )


def test_ImageIO_get_as_numpy_sucess():
    """Test getting the image as a numpy array."""
    img = ImageIO(T1_MRI)
    np_array = img.get_as_numpy()
    assert isinstance(np_array, np.ndarray)
    assert (
        np_array.shape == sitk.ReadImage(T1_MRI).GetSize()[::-1]
    )  # Reverse for numpy shape


def test_ImageIO_get_as_numpy_raise_error_no_image_loaded():
    """Test getting the image as numpy when no image is loaded."""
    img = ImageIO(image_array=np.ones((5, 5, 5)))
    img._image_as_numpy = None  # Force no image loaded
    with pytest.raises(Exception) as e:
        img.get_as_numpy()
    assert (
        e.value.args[0]
        == 'Image is not loaded as numpy array. Please load the image first.'
    )


def test_ImageIO_update_image_spacing_sucess():
    """Test updating the image spacing."""
    img = ImageIO(T1_MRI)
    new_spacing = (2.0, 2.5, 3.0)
    img.update_image_spacing(new_spacing)
    sitk_img = img.get_as_sitk()
    assert sitk_img.GetSpacing() == new_spacing


def test_ImageIO_update_image_origin_sucess():
    """Test updating the image origin."""
    img = ImageIO(T1_MRI)
    new_origin = (5.0, 10.0, 15.0)
    img.update_image_origin(new_origin)
    sitk_img = img.get_as_sitk()
    assert sitk_img.GetOrigin() == new_origin


def test_ImageIO_update_image_direction_sucess():
    """Test updating the image direction."""
    img = ImageIO(T1_MRI)
    new_direction = (1.0, 2.0, 1.1, 0.0, 1.0, 2.0, 4.0, 3.0, 1.0)
    img.update_image_direction(new_direction)
    sitk_img = img.get_as_sitk()
    assert sitk_img.GetDirection() == new_direction


def test_ImageIO_update_image_data_sucess_with_enforce_new_dimension():
    """Test updating the image data."""
    img = ImageIO(T1_MRI)
    new_data = np.random.rand(10, 10, 10)
    img.update_image_data(new_data, enforce_new_dimension=True)
    np_array = img.get_as_numpy()
    assert np.array_equal(np_array, new_data)


def test_ImageIO_save_image_sucess(tmp_path):
    """Test saving the image to a new path."""
    img = ImageIO(T1_MRI)
    save_path = tmp_path / 'saved_image.nii.gz'
    img.save_image(str(save_path))
    assert os.path.exists(save_path)
    saved_img = sitk.ReadImage(str(save_path))
    assert saved_img.GetSize() == sitk.ReadImage(T1_MRI).GetSize()


def test_ImageIO_save_image_raise_error_no_image_loaded():
    """Test saving the image when no image is loaded."""
    img = ImageIO(image_array=np.ones((5, 5, 5)))
    img._image_as_sitk = None  # Force no image loaded
    save_path = os.path.join('directory', 'not', 'found', 'saved_image.nii.gz')
    with pytest.raises(Exception) as e:
        img.save_image(str(save_path))
    assert 'The directory of the full path' in e.value.args[0]


@pytest.mark.parametrize(
    'input_data, ref_data',
    [
        (
            np.random.rand(10, 10, 10),
            ImageIO(image_array=np.random.rand(10, 10, 10)),
        ),
        (
            ImageIO(image_array=np.random.rand(10, 10, 10, 5)),
            ImageIO(image_array=np.random.rand(10, 10, 10, 5)),
        ),
        (
            ImageIO(image_array=np.random.rand(10, 10, 10, 5)).get_as_sitk(),
            ImageIO(image_array=np.random.rand(10, 10, 10, 5)),
        ),
        (
            ImageIO(image_array=np.random.rand(10, 10, 10)).get_as_ants(),
            ImageIO(image_array=np.random.rand(10, 10, 10)),
        ),
        (ImageIO(T1_MRI), ImageIO(image_path=T1_MRI)),
        (ImageIO(PCASL_MTE), ImageIO(image_path=PCASL_MTE)),
        (ImageIO(M0), ImageIO(image_path=M0)),
    ],
)
def test_check_image_properties_does_not_raises_errors_for_valid_image(
    input_data, ref_data
):
    """Test check_image_properties with a valid image."""
    check_image_properties(input_data, ref_data)
    assert True  # If no exception is raised, the test passes


def test_clone_image_sucess():
    """Test cloning an image."""
    img = ImageIO(T1_MRI)
    cloned_img = clone_image(img)
    assert isinstance(cloned_img, ImageIO)
    assert cloned_img.get_image_path() == None
    assert np.array_equal(cloned_img.get_as_numpy(), img.get_as_numpy())
    assert cloned_img.get_as_sitk().GetSize() == img.get_as_sitk().GetSize()
    assert cloned_img.get_as_ants().dimension == img.get_as_ants().dimension


def test_clone_image_sucess_with_copied_path():
    """Test cloning an image."""
    img = ImageIO(T1_MRI)
    cloned_img = clone_image(img, include_path=True)
    assert isinstance(cloned_img, ImageIO)
    assert cloned_img.get_image_path() == img.get_image_path()
    assert np.array_equal(cloned_img.get_as_numpy(), img.get_as_numpy())
    assert cloned_img.get_as_sitk().GetSize() == img.get_as_sitk().GetSize()
    assert cloned_img.get_as_ants().dimension == img.get_as_ants().dimension


def test_check_path_sucess():
    """Test check_path with a valid path."""
    valid_path = T1_MRI
    check_path(valid_path)
    assert True  # If no exception is raised, the test passes


def test_check_path_failure():
    """Test check_path with an invalid path."""
    invalid_path = os.path.join('invalid', 'path', 'to', 'image.nii.gz')
    with pytest.raises(FileNotFoundError) as e:
        check_path(invalid_path)

    assert 'The file' in e.value.args[0]

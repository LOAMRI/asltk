# Getting started

This section will help you quickly start using the `asltk` library by providing simple examples and code snippets for different scenarios.

## Simple Example of How to Use the Library

Let’s begin with a basic example that demonstrates how to import `asltk` and use its core functionality. We will start considering the basic creating of the `ASLData` object, which is the basic structure to store and manage the ASL data to use in the code:

```python
from asltk.asldata import ASLData

data = ASLData(
    pcasl='path/to/pcasl.nii.gz',
    m0='path/to/m0.nii.gz',
    ld_values=[100.0, 150.0, 250.0],
    pld_values=[100.0, 120.0, 150.0],
    te_values=[12.5, 36.7, 85.5]
)
```

The example above creates an `asldata.ASLData` object, called `data`, which will store the pCASL and M0 images, along with the LD, PLD and TE values (in this case we are showing a prototype of multiTE-ASL data loading)

!!! tip
    There are many other image file formatts that is accepted in the `asltk` tool. We adopted the `SimpleITK` API, then one can check the image formatts in the [official documentation](https://simpleitk.readthedocs.io/en/master/IO.html)

## Example operation using `asltk`

The following examples can represents some situations on ASL data processing:

1. Load a pCASL and M0 data for basic CBF/ATT mapping

```python
from asltk.asldata import ASLData


```

2.  Load a pCASL and M0 data for multiTE-ASL processing obtaingin the T1 exchange from blood to Grey Matter mapping

```python
from asltk.asldata import ASLData


```


## Load and process an image

As as standard notation, the `asltk` library assumes that all the image data files are storaged and manipulated using `numpy` objects. Therefore, the following code snippets can represents the `load_image` and `save_image` pattern adopted in the tool:

1. Loading and image

```python
from asltk import utils

img = utils.load_image('path/to/pcasl.nii.gz')
type(img)
< numpy.ndarray >
```

!!! warning
    The `asltk` uses the `SimpleITK` library to load and save images due to it's long list of image format options, e.g. NifTI, Nrrd, MHA, etc. However, in order to transpose to `numpy.array` data format for image processing, it is important to note that the image space rasteting format relies as follows:
    ```
    SimpleITK -> (x,y,z,...,n_d)
    Numpy -> (n_d, ..., z,y,x)
    ```
    Where the `n_d` represents a higher order dimension in the data.

2.  Saving and image

```python
from asltk import utils

img = utils.load_image('path/to/pcasl.nii.gz')
utils.save_image(img, 'path/to/save/image.nii.gz')
```

Basically, one can use out tool for the general pipeline:

* Loading a data and metadata, 
* Choosing the right ASL processing algorithm for the data loaded
* Saving the outcomes generated in the `asltk` tool.

## Using the library scripts

Another quick and easy way to learn how to use the ASL processing algorithms implemented in the `asltk` library is using our Python scripts.

All the scripts are located in the `asltk/scripts` folder, which is basically a simple way to run a determined kind of ASL processing.

At the moment, there are the following scripts avaliable:

1. `cbf.py`: Python script focused on calculating CBF/ATT maps using pCASL images. This script is based on the `CBFMapping` class.
2. `te_asl.py`: Python script focused on calculating T1 relaxation exchange time between Blood and Grey-Matter (T1blGM) map using pCASL multi-TE ASL images. This script is based on the `MultiTE_ASLMapping` class.

Additional scripts can be added in the `asltk` library, by the community contribution. If it is your interest, then go to the [developing scripts](contribute.md) section at the developer guidelines.
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

# Example operation using `asltk`

The following examples can represents some situations on ASL data processing:

1. Load a pCASL and M0 data for basic CBF/ATT mapping

```python
from asltk.asldata import ASLData


```

2.  Load a pCASL and M0 data for multiTE-ASL processing obtaingin the T1 exchange from blood to Grey Matter mapping

```python
from asltk.asldata import ASLData


```


# Load and process an image

As as standard notation, the `asltk` library assumes that all the image data files are storaged and manipulated using `numpy` objects. Therefore, the following code snippets can represents the `load_image` and `save_image` pattern adopted in the tool:

1. Loading and image

```python
from asltk import utils

img = utils.load_image('path/to/pcasl.nii.gz')
type(img)
< numpy.ndarray >
```

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

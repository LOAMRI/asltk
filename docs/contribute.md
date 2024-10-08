# How to Contribute

## Preparing the coding environment

The first step to start coding new features or correcting bugs in the `asltk` library is doing the repository fork, directly on GitHub, and following to the repository clone:

```bash
git clone git@github.com:<YOUR_USERNAME>/asltk.git
```

Where `<YOUR_USERNAME>` indicates your GitHub account that has the repository fork.

!!! tip
    See more details on [GitHub](https://docs.github.com/pt/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for forking a repository

After the repository been set in your local machine, the following setup steps can be done to prepare the coding environment:

!!! warning
    We assume the Poetry tool for project management, then make sure that the Poetry version is 1.8 or above. See more information about [Poetry installation](https://python-poetry.org/docs/#installing-with-pipx)

```bash
cd asltk
poetry shell && poetry install
```

Then all the dependencies will be installed and the virtual environment will be created. After all being done successfully, the shortcuts for `test` and `doc` can be called:

```bash
task test
```

```bash
task doc
```

More details about the entire project configuration is provided in the `pyproject.toml` file.

### Basic tools

We assume the following list of developing, testing and documentation tools:

1. blue
2. isort
3. SimpleITK
4. numpy
5. rich
6. scipy
7. pytest
8. taskipy
9. mkdocs-material
10. pymdown-extensions

Further adjustments in the set of tools for the project can be modified in the future. However, the details about these modifications are directly reported in new releases, regarding the specific tool versioning (more details at Version Control section)

## Code Structure

The general structure of the `asltk` library is given as the following:

``` mermaid
classDiagram
  MRIParameters <|-- CBFMapping
  MRIParameters <|-- MultiTE_ASLMapping
  class MRIParameters{
    +float T1bl
    +float T1csf
    +float T2bl
    +float T2gm
    +float T2csf
    +float Alpha
    +float Lambda
    +set_constant(value, param)
    -get_constant(param)
  }
  class CBFMapping{
    +__inti__(ASLData)
    +set_brain_mask(brain_mask, label)
    +create_map()
  }
  class MultiTE_ASLMapping{
    +__inti__(ASLData)
    +set_brain_mask(brain_mask, label)
    +set_cbf_map(cbf_map)
    -get_cbf_map()
    +set_att_map(att_map)
    -get_att_map()
    +create_map()
  }
  class ASLData{
    +NumpyArray pCASL
    +NumpyArray M0
    +List LD
    +List PLD
    +List TE (optional)
    +List DW (optional)
  }
```

Where the `ASLData` class informs the basic data structure to all the ASL mapping strategies. All the ASL processing methods, implemented in the `reconstruction.py` module, have to include a `ASLData` object to the class instance. In particular, for each ASL processing instance that has been created, one can change it's particular `MRIParameters` calling the inheritance parameters.

!!! note
    The `ASLData` need to provide at least the LD, PLD sequences and the pCASL and M0 images. The rest of information is only needed depending on the ASL imaging technique that is provided. 

The implementation of `CBFMapping` class already offers both the `cbf` and `att` maps, given by the Buxton model [^1]. In general, the same advanced ASL imaging protocols can offer a possibility to calculate the `cbf` and `att` maps, using a specific part of the input data. This is the case for the `MultiTE_ASLMapping` class, which uses internally the `CBFMapping` output to proceed to the `MultiTE_ASLMapping` calculation. 

The `MultiTE_ASLMapping` class provides the implementation of the multi time of echos (TE) ASL imaging technique [^2]

!!! tip
    If the CBF and ATT maps are already present before the call for `MultiTE_ASLMapping` calculation, then the `set_cbf_map` and `set_att_map` methods are useful.


As a coding patterns, it is expected that new ASL processing techniques will be placed in the class organization given as above diagram. All new classes that can be added in the future, should be as follows:

1. A new class implementation in the `reconstruction.py` module
2. Provides a collection of `set` and `get` methods for required additional data that is not responsability to the `ASLData` structure
3. Implements the `create_map()` method, giving the correct input parameters required in the specific ASL processing method.

!!! question
    In case of any doubt, discuss with the community using a [issue card](https://github.com/LOAMRI/asltk/issues) in the repo.

[^1]: R B Buxton, et al. "A general kinetic model for quantitative perfusion imaging with arterial spin labeling". Magn Reson Med (1998). PMID: 9727941 DOI: 10.1002/mrm.1910400308

[^2]: L Petitclerc et al. "Ultra-long-TE arterial spin labeling reveals rapid and brain-wide blood-to-CSF water transport in humans". Neuroimage (2022).  DOI: 10.1016/j.neuroimage.2021.118755

## Testing

Another coding pattern expected in new contributions in the `asltk` library is the uses of unit tests. 

!!! info
    A good way to implement test together with coding steps is using a Test-Driven Desing (TDD). Further details can be found at [TDD tutorial](https://codefellows.github.io/sea-python-401d2/lectures/tdd_with_pytest.html) and in many other soruces on internet

Each module or class implemented in the `asltk` library should have a series of tests to ensure the quality of the coding and more stability for production usage. We adopted the Python `codecov` tool to help in collecting the code coverage status, which can be accessed by the HTML page that is generated on the call

```bash
task test
```

## Code Documentation

The coding documentation pattern is the [Google Docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

Please, provide as much details as possible in the methods, classes and modules implemented in the `asltk` library. By the way, if one may want to get deeper in the explanation of some functionality, then use the documentation webpage itself, which can be easier to add figures, graphs, diagrams and much more simple to read.

!!! tip
    As a good form to assist further users is providing `Examples` in the Google Docstring format. Then, when it is possible, add a few examples in the code documentation. 

!!! info
    The docstring also passes to a test analysis, then take care about adding `Examples` in the docstring, respecting the same usage pattern for input/output as the code provides



## Version Control

The `asltk` project adopts the [Semantic Versioning 2.0.0 SemVer](https://semver.org/) versioning pattern. Please, also take care about the specific version changes that will be added by further implementations.

Another important consideration is that the `asltk` repository has two permanent branches: `main` and `develop`. The `main` branch is placed to stable, versioning controled releases, and the `develop` branch is for unstable most up-to-date functionalities. In order to keep the library as more reliable as possible, please consider making a Pull Request (PR) at the `develop` branch before passing it to the `main` branch.


## Extending the library

### Extending core functionalities

If you want to provide a new functionality in the `asltk`, e.g. a new class that supports a novel ASL processing method, please keep the same data and coding structure as described in the `Code Structure` section.

Any new ideas to improve the project readbility and coding organization is also welcome. If it is the case, please raise a new issue ticket at GitHub, using the Feature option to open an community debate about your suggestion. Once it is approved, a new project version is release with the new implementations glued in the core code.

### Scripts

A easier and less burocratic way to provide new code in the project is using a Python script. In this way, a simple calling script can be added in the repository, under the `scripts` folder, that can be used directly using the python command:

```bash
python -m asltk.scripts.YOUR_SCRIPT [input options]
```

In this way, you can share a code that can be called for a specific execution and can be used as a command-line interface (CLI). There are some examples already implemented in the `asltk.scripts`, such as the `cbf.py` and `te_asl.py`, which calculates the CBF/ATT and the T1 relaxation exchange between blood-GM, respectively.

For instance, a call execution considering the `cbf.py` script can be showed using the `--help` option:

```bash
usage: cbf.py [-h] --pld PLD [PLD ...] --ld LD [LD ...] [--verbose] pcasl m0 [mask] [out_folder]

Python script to calculate the basic CBF and ATT maps from ASL data.

positional arguments:
    pcasl                ASL raw data obtained from the MRI scanner. This must be the basic PLD ASL MRI acquisition protocol.
    m0                   M0 image in Nifti format.
    mask                 Image mask defining the ROI where the calculations must be done. Any pixel value different from zero will be assumed as the ROI area.
                        Outside the mask (value=0) will be ignored.
    out_folder           The output folder that is the reference to save all the output images in the script. The images selected to be saved are given as tags
                        in the script caller, e.g. the options --cbf_map and --att_map. By default, the TblGM map is placed in the output folder with the name
                        tblgm_map.nii.gz

    options:
    -h, --help           show this help message and exit
    --pld PLD [PLD ...]  Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers
    --ld LD [LD ...]     Labeling Duration trend (LD), arranged in a sequence of float numbers.
    --verbose            Show more details thoughout the processing.
```

!!! tip
    Feel free to get inspired adding new scripts in the `asltk` project. A quick way to get this is simply making a copy of an existing python script, e.g. the `cbf.py` script, and making your specific modifications.

!!! info
    We adopted the general Python `Argparse` scripting module to create a standarized code. More details on how to use it can be found at the [official documentation](https://docs.python.org/3/library/argparse.html)

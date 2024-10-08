# Installation guide

## Prerequisites

Before installing `asltk`, ensure you have the following: 

* Python version 3.10 or higher:

You can verify your Python version by running:
```bash
python --version
```

* pip: 
 
Ensure you have pip (Python's package installer) up to date. You can check this by running:

```bash
pip --version
```

If you need to upgrade pip, run:

```bash
python -m pip install --upgrade pip
```


## Installing the Library

The `asltk` package is available on PyPI and can be installed easily using pip. To install the latest stable version, simply run:

!!! note
    Even thought the `asltk` installation examples are assuming the `pip` command in Linux terminal commands, the tool is also able to be installed using other Python library managers, e.g. Conda or Poetry, and also being able to run on MacOS and Windows operating systems.

!!! info
    Please be aware that the installation procedure of any Python library should be properly done using a virtual environment. Hence, in this manner, it is indicated to also proceed with the following command:
    ```bash
    python3 pip install venv
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Recall that this procedure is exemplified using a Linux system. A general instruction about the virtual environment can be found at [venv python documentation page](https://docs.python.org/3/library/venv.html). 

```bash
pip install asltk
```

## Verifying the Installation

After the installation process has been completed, you can verify that the library is correctly installed by importing it in Python:

```bash
pip freeze | grep asltk
```

If the installation was successful, this command should print the version number of `asltk`.
# Development Process

The `asltk` project is always able to receive new contributors, which greatly helps to keep it updated, more secure and with the most recent advances in the ASL knowledge field.

However, in order to maintain a development standard, we must to inform a set of (relativaly) simple tools, process and coding patterns.

Please follow this tutorial to prepare your system environment and understand the general coding concepts adopted in the ASL toolkit project.

## Preparing the development enviroment

First, clone the repo from GitHub.

> [!IMPORTANT]
> It is assumed that the user is located at a folder where it has permission to read/write files. 

As suggestion, create a folder at your home directory  

```bash
cd $HOME && mkdir git-projects && cd git-projects
```

Then, at the `git-projects` folder, clone the repo:

```bash
git clone git@github.com:LOAMRI/asltk.git
```

After the clone been done, one can create a virtual environment to assist the Python installation without messing up with the global system configuration.

> [!NOTE]
> We assume the usage of `venv` tool here. But the user can choose any other python virtual environment

```
sudo apt-get install python3-venv
python3 -m venv .venv
source .venv/bin/activate
```

Once the `venv` is activated, the `asltk` library installation can be performed simply by:


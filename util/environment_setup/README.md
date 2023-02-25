# Environment Installation

## Using Anaconda
- Install a suited version of Anaconda according to [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html) if it is not already installed
- Use either [conda_requirements_linux.txt](conda_requirements_linux.txt) or [conda_requirements_windows.txt](conda_requirements_windows.txt). There are different versions for the operatins systems
since combinations of some packages caused problems on a specific system. The provided requirements are tested on the respective OS.
```
conda install --file requirements.txt
```

## Using pip
- Install a suited pip version if it is not already installed
- Use either [conda_requirements_linux.txt](conda_requirements_linux.txt) or [conda_requirements_windows.txt](conda_requirements_windows.txt). There are different versions for the operatins systems
since combinations of some packages caused problems on a specific system. The provided requirements are tested on the respective OS.
```
pip install -r requirements.txt
```
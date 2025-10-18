# AI in classification of understudied bird species using the sound they produce

> **Note:** All the commands are based on a Unix based system such as _OSX_.
> For a different system look for similar commands for it.


## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```

### Python virtual environment

**Create** a virtual environment:

```bash
$ python3 -m venv .ecg
```

`.ecg` is the name of the folder that would contain the virtual environment.

**Activate** the virtual environment:

```bash
$ source .ecg/bin/activate
```

**Windows**
```bash
source .ecg/Scripts/activate
```
### Requirements

```bash
(.venv) $ pip install -r requirements.txt
```

1. Fill in the values appropriately

## Run the queries

Open a **jupyterlab** instance

```bash
$ jupyter-lab
```

The code should be present in the `spectrogram-method.ipynb` and `rawWave-method.ipynb` files.

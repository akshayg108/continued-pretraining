# continued-pretraining

A benchmark and toolkit for continued pretraining methods including LeJEPA, SimCLR, MAE, DIET, and domain adaptation baselines (TENT, ...).

## Installation

### Option 1: Using Poetry (Recommended)

If you have [Poetry](https://python-poetry.org/) installed, it can automatically install git dependencies:

```bash
poetry install
```

Poetry will automatically clone and install `stable-pretraining` and `stable-datasets` from GitHub.

### Option 2: Using pip

Before installing this package with pip, you must first install `stable-pretraining` and `stable-datasets` from their GitHub repositories:

```bash
# Install stable-pretraining
pip install git+https://github.com/galilai-group/stable-pretraining.git

# Install stable-datasets
pip install git+https://github.com/galilai-group/stable-datasets.git
```

Then install this package in development/editable mode:

```bash
pip install -e .
```

This installs the `stable-cp` package along with all other dependencies defined in `pyproject.toml`.

### Dependencies

The package requires:
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Lightning ≥ 2.0.0
- transformers ≥ 4.30.0
- Other dependencies listed in `pyproject.toml`


## Development Setup

### Contributing to stable-pretraining and/or stable-datasets

If you want to contribute to `stable-pretraining` and/or `stable-datasets` alongside this project, install them in editable mode:

#### Using Poetry

```bash
cd ..  # Go to parent directory

# Clone stable-pretraining and stable-datasets
git clone https://github.com/galilai-group/stable-pretraining.git
git clone https://github.com/galilai-group/stable-datasets.git

cd continued-pretraining

# Add local editable dependencies
poetry add --editable ../stable-pretraining
poetry add --editable ../stable-datasets
poetry install
```

#### Using pip

```bash
cd ..  # Go to parent directory

# Clone and install stable-pretraining in editable mode
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
pip install -e .
cd ..

# Clone and install stable-datasets in editable mode
git clone https://github.com/galilai-group/stable-datasets.git
cd stable-datasets
pip install -e .
cd ..

# Install continued-pretraining in editable mode
cd continued-pretraining
pip install -e .
```

## Package Structure

After installation, you can import the package:

```python
from stable_cp.methods.lejepa.lejepa_cp import setup_lejepa
from stable_cp.methods.simclr.simclr_cp import setup_simclr
from stable_cp.methods.mae.mae_cp import setup_mae_cp
from stable_cp.methods.diet.diet_cp import setup_diet
from stable_cp.methods.tent.tent_cp import setup_tent_cp
from stable_cp.callbacks import FreezeBackboneCallback
from stable_cp.evaluation.zero_shot_eval import zero_shot_eval
```

## Usage

Run continued pretraining experiments:


```bash
python continued_pretraining.py --method lejepa --dataset <dataset> --backbone <model>
```

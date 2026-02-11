# continued-pretraining

A benchmark and toolkit for continued pretraining methods including LeJEPA, SimCLR, MAE, DIET, and domain adaptation baselines (TENT, ...).

## Installation

### Standard Installation

Install the package in editable mode:

```bash
pip install -e .
```

This will automatically install all dependencies including `stable-pretraining` and `stable-datasets` from their GitHub repositories.

**Note:** The first installation may take a few minutes as pip clones and installs the git dependencies.

### Dependencies

The package requires:
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Lightning ≥ 2.0.0
- transformers ≥ 4.30.0
- Other dependencies listed in `pyproject.toml`


## Development Setup

### Contributing to stable-pretraining and/or stable-datasets

If you want to contribute to `stable-pretraining` and/or `stable-datasets` alongside this project, install them in editable mode first:

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
# pip will use the already-installed editable versions above
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

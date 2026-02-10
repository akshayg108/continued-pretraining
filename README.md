# continued-pretraining

A benchmark and toolkit for continued pretraining methods including LeJEPA, SimCLR, MAE, DIET, and domain adaptation baselines (TENT, ...).

## Installation

### Standard Installation (Editable Mode)

Install the package in development/editable mode from the repository root:

```bash
pip install -e .
```

This installs the `stable-cp` package along with all dependencies defined in `pyproject.toml`.

### Dependencies

The package requires:
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Lightning ≥ 2.0.0
- transformers ≥ 4.30.0
- Other dependencies listed in `pyproject.toml`


## Development Setup

### With stable-pretraining

If you want to contribute to `stable-pretraining` alongside this project:

```bash
cd ..  # Go to parent directory
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
pip install -e .
cd ../continued-pretraining
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

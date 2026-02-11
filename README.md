# continued-pretraining

A benchmark and toolkit for continued pretraining methods including LeJEPA, SimCLR, MAE, DIET, and domain adaptation baselines (TENT, ...).

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

This will automatically install all dependencies including `stable-pretraining` and `stable-datasets` from their GitHub repositories.

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

## Usage

Run continued pretraining experiments:

```bash
python continued_pretraining.py --method lejepa --dataset <dataset> --backbone <model>
```

# Scripts

Utility scripts for SWE4DVar development and deployment.

## verify_environment.py

Verify that all required dependencies are installed:

```bash
python scripts/verify_environment.py
```

## Environment Setup

To create the conda environment:

```bash
conda env create -f environment.yml
conda activate swe4dvar
pip install -e .
```

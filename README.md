# GrassST

## Install Environment

Create the GrassST conda environment:

```bash
conda env create -f envs/environment_grassst.yml
```

Activate the environment:

```bash
conda activate grassST_env
```

To update an existing environment:

```bash
conda env update -f envs/environment_grassst.yml --prune
```

## Test GrassST

Run the GrassST test script:

```bash
python test/test05011_grass5.py
```

The script runs the GrassST framework using the dataset paths configured inside `test/test05011_grass5.py`.

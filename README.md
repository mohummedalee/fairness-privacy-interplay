To set up, run

```
conda env create -f environment.yml
conda activate fairness-privacy
python -m ipykernel install --user --name=fairness-privacy
```

The first two lines set up a conda environment, the third one registers it to ipykernel so it can be used in Jupyter Lab.

Note that this `environment.yml` file is optimal for running on a Mac since it installs the nightly version of PyTorch with MPS acceleration. There's a separate version for running on Discovery found under `/work/eai/muh.ali/fairness-privacy/environment.yml` on the cluster.
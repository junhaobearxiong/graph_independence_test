# Graph Independence Testing

This repository contains the code for running the experiments in the manuscript: [Xiong, Junhao, et al. “Graph Independence Testing.” arXiv preprint arXiv: 1906.03661 (2019)](https://arxiv.org/abs/1906.03661). The manuscript is currently under major revision, so is the code, so you may not find the exact code to reproduce the figures in the manuscript.

## Files

The main functions with the core functionalities are in `core.py`, which contains function to compute test statistics, p-values, powers, etc.


### Simulations

`simulations.py` contains function to simulate $\rho$-correlated Bernoulli SBMs, $\rho$-correlated Bernoulli DC-SBMs and $\rho$-correlated Gaussian SBMs (based on `graspologic` implementation but are more general)

The following files correspond (roughly) to figures in the manuscript: 
1. `experiments/sim_teststats.py` and `plotting/plot_sim_test_statistic.py` are used to generate Figure 1.
2. `experiments/sim_power.py` and `plotting/plot_sim_power.py` are used to generate Figure 3 and 4.


### Real data experiments

This directory currently contains the code to run experiment on the the following datasets:
1. `mouse`: a [dataset](https://github.com/v715/popcon/tree/master/popcon/datasets/data/duke) containing connectomes of 4 different species of mice
2. `timeseries`: a dataset containing the connectome of a single subject sequenced over many time points in time
3. `cpac200`
4. `enron`

To run experiments on the associated dataset, here is a standard workflow:
1. Preprocess the raw dataset into a `numpy.array` with the following format: [# graphs, # vertices, # vertices]. You may need to write parts of this, but it should be straightforward using the functions available in `data_utils`.
2. (optional) Apply a transformation to the graphs using `experiments/real_transfrom_data.py` 
3. (optional) Estimate community assignments of the graphs using `experiments/real_community_estimation.py`
4. Run `experiments/real_teststats_pval.py` with the appropriate command-line arguments

## Setup

Install Python 3.6. You can use `pyenv` to manage the Python versions on your machine.

Next, set up the local environment in the `./venv` directory:

```
python -m venv ./venv
``` 

To activate the environment, type:
```
. venv/bin/activate
```

Then, install the requirements in the local environment:
```
pip install --upgrade pip
pip install -r requirements.txt
```

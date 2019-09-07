# Graph Independence Testing

This repository contains the code for the experiments in the manuscript: Xiong, Junhao, et al. “Graph Independence Testing.” arXiv preprint arXiv: 1906.03661 (2019). Please go to directory v1 for the most recent version.

# Files

1. `simulations.py` contains function to simulate $\rho$-correlated Bernoulli SBMs and $\rho$-correlated Gaussian SBMs.
2. `utils.py` contains various functions to compute power, perform block permutations, estimate block assignments, etc.
3. `sim_test_stats.py` and `plot_sim_test_stats.py` are used to generate Figure 1.
4. `sim_power_bernoulli.py` and `plot_sim_power_bernoulli.py` are used to generate Figure 3.
5. `sim_power_gaussian.py` and `plot_sim_power_gaussian.py` are used to generate Figure 4.
6. `real_celegans.py` and `plot_real_celegans.py` are used to generate Figure 7.

# Setup

Install Python 3.6. You can use `pyenv` to manage the Python versions on your machine.

Next, enter directory v1. set up the local environment in the `./env` directory.

```
python -m venv ./env
``` 

To activate the environment, type:
```
. env/bin/activate
```

Then, install the requirements in the local environment:
```
pip install --upgrade pip
pip install -r requirements.txt
```

Note, for the modules `mgcpy` and `graspy`, since they are in active development in the time the manuscript was written, a stable release version was not available for all the necessary functionalities. Therefore, you can clone both of these repositories on your local machine, outside this repository, then update your `PYTHONPATH` variable to include the directory where you include these repositories. For example, you can add:
```
export PYTHONPATH=${PYTHONPATH}:$HOME/Projects/
``` 
in the `.bashrc` file, if you install these repositories in the directory `~/Projects`. 
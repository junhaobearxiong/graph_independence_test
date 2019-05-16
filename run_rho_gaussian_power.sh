#!/bin/bash

PYTHONPATH=/Users/jxiong/Documents/Projects/mgcpy:$(pwd)
python bin/rho_gaussian_sbm_power.py 500 er
python bin/rho_gaussian_sbm_power.py 500 er_marg
python bin/rho_gaussian_sbm_power.py 500 sbm
python bin/rho_gaussian_sbm_power.py 500 sbm_marg

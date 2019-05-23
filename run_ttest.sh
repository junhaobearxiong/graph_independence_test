#!/bin/bash

PYTHONPATH=/Users/jxiong/Documents/Projects/mgcpy:$(pwd)
python bin/ttest.py 500 er
python bin/ttest.py 500 er_marg
python bin/ttest.py 500 sbm
python bin/ttest.py 500 sbm_marg

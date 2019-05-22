#!/bin/bash

PYTHONPATH=/Users/jxiong/Documents/Projects/mgcpy:$(pwd)
python bin/celegans_chem_gap.py 500 1 0
python bin/celegans_chem_gap.py 500 1 1
python bin/celegans_chem_gap.py 500 1 2

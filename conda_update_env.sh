#!/bin/bash

conda env update -f ./environment.yaml
conda env export -p ./env > ./environment.yaml

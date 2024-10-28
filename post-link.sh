#!/bin/bash

# Create the activate.d and deactivate.d directories
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Create the activation script for setting CUDA_HOME
echo 'export CUDA_HOME=$CONDA_PREFIX' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Create the deactivation script to unset CUDA_HOME
echo 'unset CUDA_HOME' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

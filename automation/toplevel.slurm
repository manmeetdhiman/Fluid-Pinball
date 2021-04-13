#!/bin/bash

#SBATCH --job-name=GA_run_4
#SBATCH --time=7-00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=12
#SBATCH --mem=0
#SBATCH --mail-user=richard.gao@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=single

module load ~/software/miniconda3/bin/python
which python
python --version
#conda create -n env
#conda install -n env pytorch
#conda install -n env matplotlib
#conda install -n env pyyaml
#conda install -n env scipy
#conda install -n env pandas
source activate env
python -u toplevel.py > main.out

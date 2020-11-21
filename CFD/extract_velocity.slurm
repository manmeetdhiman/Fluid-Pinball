#!/bin/bash

#SBATCH --job-name=pinball_extract_velocity
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0
#SBATCH --mail-user=richard.gao@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=cpu2019

module load python/anaconda-3.6-5.1.0
python3 -u mp_extract_velocity.py > extract_velocity_py.out

#!/bin/bash

#SBATCH --job-name=RL-fix
#SBATCH --time=7-00:00
#SBATCH --mem=0
#SBATCH --mail-user=richard.gao@ucalgary.ca
#SBATCH --partition=single

echo "Starting Python job started at: `date`"

module load python/anaconda-3.6-5.1.0
python3 -u fix_errors.py > errors.out

echo "Python job finished at: `date`"

#!/bin/bash

#SBATCH --job-name=pinball
#SBATCH --account=def-martinuz
#SBATCH --time=03-00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=pinball.out
#SBATCH --mail-user=richard.gao@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

INPUT=pinball_v4.jou
OUTPUT=pinball.out
SOLVER="2ddp"

module load ansys/19.2

FLUENT=`which fluent`
echo "Using Fluent: $FLUENT"

echo "Current working directory is `pwd`"

slurm_hl2hl.py --format ANSYS-FLUENT > machinefile
NCORE=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))

echo "Starting Fluent job at: `date`"

$FLUENT $SOLVER -g -t $NCORE -cnf=machinefile -mpi=intel -affinity=0 -i $INPUT > $OUTPUT>&1

echo "Fluent Job finished at: `date`"

#!/bin/bash

#SBATCH --job-name=pinball
#SBATCH --time=96:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=richard.gao@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=parallel

INPUT=pinball_v4.jou
OUTPUT=pinball.out
SOLVER="2ddp"

module load ansys/2019r2

FLUENT=`which fluent`
echo "Using Fluent: $FLUENT"

echo "Current working directory is `pwd`"

# Create a node list so that Fluent knows which nodes to use.
HOSTLIST=hostlist_${SLURM_JOB_ID}
scontrol show hostnames > $HOSTLIST
echo "Created host list file $HOSTLIST"
echo "Running on hosts:"
cat $HOSTLIST

echo "Using $SLURM_NTASKS cores."

echo "Starting run at: `date`"

$FLUENT $SOLVER -g -t${SLURM_NTASKS} -ssh -cnf=${HOSTLIST} -i $INPUT > $OUTPUT 2>&1

echo "Job finished at: `date`"

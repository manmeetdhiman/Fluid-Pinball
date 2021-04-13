#!/bin/bash

#SBATCH --job-name=pinball
#SBATCH --output=worker.out
#SBATCH --time=7-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=0
#SBATCH --mail-user=richard.gao@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=parallel
##SBATCH --account=def-martinuz


# USER INPUTS
ID=$1
if [ -z $ID ]; then
	echo "Bash script needs ID!"
  echo "Exiting"
  exit
else
	echo "--------------------------------------------"
	echo "WORKER $ID"
	echo "--------------------------------------------"
fi

CLUSTER="ARC" # "ARC", "CC"
ALGORITHM_TYPE="GA"
HOME_PATH="/home/richard.gao/${ALGORITHM_TYPE}_production_run_4/Results_${ALGORITHM_TYPE}"
NEW_CWD=$HOME_PATH
SCRIPT_PATH="/home/richard.gao/${ALGORITHM_TYPE}_production_run_4/CFD/data_functions.py"
JOB_FILE="job"
WORKER_FILE="worker${ID}_${SLURM_JOB_ID}"
SLEEP_TIME=30
# Fluent inputs
INPUT=pinball_v4.jou
OUTPUT="pinball.out"
SOLVER="2d"
N_SIMULATIONS=5000  # how many simulations can this script run before hitting time limit

function run_fluent ()
{
  echo "--------------------------------------------"

  if [ $CLUSTER == "CC" ]; then
    module load ansys/19.2

    FLUENT=`which fluent`

    echo "Current working directory is `pwd`"

    slurm_hl2hl.py --format ANSYS-FLUENT > machinefile
    NCORE=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))

    echo "Starting Fluent job at: `date`"

    $FLUENT $SOLVER -g -t $NCORE -cnf=machinefile -mpi=intel -affinity=0 -i $INPUT > $OUTPUT>&1

    echo "Fluent Job finished at: `date`"
  elif [ $CLUSTER == "ARC" ]; then
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

    echo "Starting Fluent job at: `date`"

    $FLUENT $SOLVER -g -t${SLURM_NTASKS} -ssh -pib -cnf=${HOSTLIST} -i $INPUT > $OUTPUT 2>&1

    echo "Fluent job finished at: `date`"
  fi

  echo "--------------------------------------------"
}

# Read in text file and assign as cwd
function read_cwd ()
{
  echo "--------------------------------------------"
  echo "Reading $1"
  dos2unix $1

  while IFS= read -r line
  do
    NEW_CWD="$line"
    break
  done < "$1"

  echo "Done reading $1. Removing it..."
  rm -rf $1

  echo "CWD: `pwd`"
  echo "Changing to: $NEW_CWD"
  cd $NEW_CWD
  echo "CWD: `pwd`"
  echo "--------------------------------------------"
}

# Run Python data extraction script
function run_python ()
{
  echo "--------------------------------------------"
	# Make sure we only run Python if all the ascii files are present
	IFS='-' read -ra ADDR <<< "$PWD"  # splits the CWD path using '-'.
	# e.g. /home/richard.gao/GA_production_run_2/Results_GA/individual-100-0-2000
	LAST_TIMESTEP=${ADDR[-1]}  # the last split is the ending timestep (e.g. 2000)

	if ! compgen -G "./ascii${LAST_TIMESTEP}" > /dev/null; then
		echo "ascii${LAST_TIMESTEP} not found. No need to run Python."
	elif compgen -G "./sensor_data.yaml" > /dev/null; then
		echo "Sensor_data.yaml file already exists. No need to run Python."
	else
	  if [ $CLUSTER == "CC" ]; then
	    echo "Starting Python job started at: `date`"

	    module load python/3.6
	    module load scipy-stack
	    virtualenv --no-download $SLURM_TMPDIR/env
	    source $SLURM_TMPDIR/env/bin/activate
	    pip install --no-index --upgrade pip
	    pip install PyYAML --no-index
	    python -u $SCRIPT_PATH $PWD > ${PWD}/extract_velocity.out

	    echo "Python job finished at: `date`"
	  elif [ $CLUSTER == "ARC" ]; then
	    echo "Starting Python job started at: `date`"

	    module load python/anaconda-3.6-5.1.0
	    python3 -u $SCRIPT_PATH $PWD > ${PWD}/extract_velocity.out

	    echo "Python job finished at: `date`"
	  fi
	fi

  echo "--------------------------------------------"
}

# Main Loop
for ((sim=1; sim <= $N_SIMULATIONS; sim++))
do
  echo "--------------------------------------------"
  echo "Home Path worker $ID: $HOME_PATH"
  cd $HOME_PATH

	sleep $ID  # this is to offset the sleep timers so two workers dont open the same file
  # Wait until job files exists
	while :
	do
	  for file in "$HOME_PATH"/*;
	  do
	   if [[ "$file" == *"$JOB_FILE"* ]]; then
			 echo "--------------------------------------------"
			 echo "JOB: $file"
			 echo "--------------------------------------------"
			 # ------------------------------------
			 # Change directory to desired
			 # ------------------------------------
	     read_cwd "$file"
	     break 2
	   fi
	  done
	  echo "Waiting for jobs"
	  echo "Sleeping for $SLEEP_TIME s"
	  sleep $SLEEP_TIME
	done

  # ------------------------------------
  # Run Fluent simulation
  # ------------------------------------
	if compgen -G "./sensor_data.yaml" > /dev/null; then
		echo "Sensor_data.yaml file already exists. No need to run Fluent."
	elif ls worker* > /dev/null 2>&1; then
		echo "A worker is already running this simulation."
	elif ! compgen -G "${INPUT}" > /dev/null; then
		echo "${INPUT} not found."
	else
		echo $ID > $WORKER_FILE
		while :
		do
			# Loop until ascii files exist. This shows that the simulation finished.
			# Need this loop because Fluent may crash/not start due to unavailable licenses.
			# Make sure we only run Fluent if ending timestep ascii doesn't exist
			IFS='-' read -ra ADDR <<< "$PWD"  # splits the CWD path using '-'.
			# e.g. /home/richard.gao/GA_production_run_2/Results_GA/individual-100-0-2000
			LAST_TIMESTEP=${ADDR[-1]}  # the last split is the ending timestep (e.g. 2000)

			if ! compgen -G "./ascii${LAST_TIMESTEP}" > /dev/null; then
				echo "ascii${LAST_TIMESTEP} not found. Running Fluent for worker $ID..."
				run_fluent
				# Fail safe. If the script is in the wrong directory, can manually create a
				# job_worker${ID} to force change the script's directory.
				if [ -f "job_worker${ID}.txt" ]; then
					echo "New job_worker${ID}.txt detected. Changing directory..."
					read_cwd "job_worker${ID}.txt"
				fi
			else
				echo "Ascii files exist"
				echo "Fluent job finished successfully"
				break
			fi
			sleep $SLEEP_TIME
		done
		# ------------------------------------
		# Extract Fluent data
		# ------------------------------------
		run_python

	fi

done

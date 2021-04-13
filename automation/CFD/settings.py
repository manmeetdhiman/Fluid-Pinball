# -------------------------------------------
# General settings
# -------------------------------------------
ALGORITHM_TYPE = 'GA'  # RL/GA
CLUSTER = 'ARC'  # CC/ARC

# -------------------------------------------
# Automation settings
# -------------------------------------------
N_WORKERS = 9
WORKER_JOB_FILE = 'job'  # this is the JOB FILE that is given to the worker.

# How often do we check if sims are finished
SLEEP_TIME = 60  # s
EXTRACT_DATA_SLEEP_TIME = 15  # s

# -------------------------------------------
# cfd.py settings
# -------------------------------------------
#COPY_COMMAND = 'powershell.exe cp'  # for testing on windows
COPY_COMMAND = 'cp'

# HOME_PATH = '/Users/richard/Desktop/Capstone/python/automation'
HOME_PATH = '/home/richard.gao/GA_production_run_4'
INIT_FILE_PATH = f'{HOME_PATH}/sim_files'
RESULTS_FOLDER_NAME = f'Results_{ALGORITHM_TYPE}'
RESULTS_FOLDER_PATH = f'{HOME_PATH}/{RESULTS_FOLDER_NAME}'
if ALGORITHM_TYPE == 'GA':
    ITERATION_FOLDER_NAME = 'generation'
    INDIVIDUAL_FOLDER_NAME = 'individual'
elif ALGORITHM_TYPE == 'RL':
    ITERATION_FOLDER_NAME = 'iteration'
    INDIVIDUAL_FOLDER_NAME = 'action'
WORKER_SLURM_FILE = 'worker.slurm'
FLUENT_JOB_FILE = 'fluent.slurm'
FLUENT_JOU_FILE = 'pinball_v4.jou'
REQUIRED_SIM_FILES = [
    FLUENT_JOB_FILE,
    FLUENT_JOU_FILE,
]
T_START_NAME = 't_start.txt'
MOTOR_ROTATIONS_FILE = 'motor_rotations.txt'
INLET_VELOCITY_FILE = 'inlet_velocity.txt'
MOTOR_PARAMS_FILENAME = 'motor_params'

# CFD simulation settings
D = 0.02  # m
R = D/2
U_INF = 1.5  # m/s
TIMESTEP_INTERVAL = 5
TOTAL_TIMESTEPS = 2000

# Cost function settings
GAMMA_COST_FACTOR = 0.02


# -------------------------------------------
# data_functions.py settings
# -------------------------------------------
# Sensor Locations
SENSOR_LOCATIONS = {
    'top': (5*D, 1.25*D),
    'mid': (5*D, 0*D),
    'bot': (5*D, -1.25*D),
}

SENSOR_DATA_YAML_FILE = 'sensor_data.yaml'
SENSOR_DATA_JSON_FILE = 'sensor_data.json'



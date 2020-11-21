import os
import time
import numpy as np
import math
import data_functions

## USER INPUTS ##
copy_command = 'powershell.exe cp'
# copy_command = 'cp'

HOME_PATH = '/home/richard.gao/capstone/sim5-automation-script-test/Sim_Files/'  # init file location
FLUENT_JOB_FILE = 'fluent.slurm'
PYTHON_JOB_FILE = 'python.slurm'
FLUENT_JOU_FILE = 'pinball_v2.jou'
REQUIRED_SIM_FILES = [
    FLUENT_JOB_FILE,
    FLUENT_JOU_FILE,
]
RESULTS_FOLDER_NAME = 'Results2'
RESULTS_FOLDER_PATH = '/home/richard.gao/capstone/sim5-automation-script-test/' + RESULTS_FOLDER_NAME
ITERATION_FOLDER_NAME = 'iteration-'
INDIVIDUAL_FOLDER_NAME = 'individual-'
T_START_NAME = 't_start.txt'

INTERVAL = 5  # timestep interval
SHEDDING_CYCLE = 5  # number of timesteps per shedding cycle

# CFD simulation settings
D = 0.02  # m
R = D/2
U_INF = 1.5

# Cost function settings
GAMMA_COST_FACTOR = 0.02


SLEEP_TIME = 30  # s

def create_omega_txt(amp, freq, phase, offset, t_end, path, dt=5e-4):
    # IMPORTANT

    # If you give a sim t_start = 0 and t_end = 4.
    # Output.txt gives t = 1, 2, 3, 4 data. NO data for t = 0.
    # Therefore, t-omega txt file should only have data for 1 2 3 4. NOT 0 1 2 3 4.
    # Otherwise, sim will use the first 4 rows (e.g. 0 1 2 3) instead of the correct (1 2 3 4)

    # Assumes phase in RADIANS.
    motor_speeds = [
        lambda t: amp[0]*math.sin(freq[0]*t + phase[0]) + offset[0],
        lambda t: amp[1]*math.sin(freq[1]*t + phase[1]) + offset[1],
        lambda t: amp[2]*math.sin(freq[2]*t + phase[2]) + offset[2]
    ]
    with open(path + 'motor_parameters.txt', 'w') as f:
        for i in range(1, t_end + 1):
            t = i*dt
            w1 = motor_speeds[0](t)
            w2 = motor_speeds[1](t)
            w3 = motor_speeds[2](t)
            f.write(f'{i} {w1: 0.5f} {w2: 0.5f} {w3: 0.5f}\n')


def create_inlet_velocity(amp, freq, phase, offset, base, t_start, n_periods, t_end, dt=5e-4):
    # Assumes phase in RADIANS.
    v = lambda t: amp*math.sin(freq*t + phase) + offset
    t_period = ( 2*math.pi - phase ) / (freq)
    n = round(t_period * n_periods / dt)

    with open('inlet_velocity.txt', 'w') as f:
        counter = 0
        for i in range(1, t_end + 1):
            if t_start <= i <= t_start + n:
                v1 = abs(v(counter*dt)) + base  # abs of sine function we cant have neg u-inf
                f.write(f'{i} {v1} \n')
                counter += 1
            else:
                f.write(f'{i} {base} \n')


def get_t_start_end_at_n(n):
    # Iteration must start at 1
    t_start = (n-1)*SHEDDING_CYCLE
    t_end = t_start + SHEDDING_CYCLE
    return t_start, t_end


def get_individual_folder_name(individual, t_start, t_end):
    return INDIVIDUAL_FOLDER_NAME + str(individual) + '-' + str(t_start) + '-' + str(t_end)


def run_cfd_individual(t_start, t_end, path, individual, motor_params, submit_job=False):
    print(f'Creating CFD folder for individual {individual}-{t_start}-{t_end} in folder {path}')
    folder_name = get_individual_folder_name(individual, t_start, t_end)
    working_path = path + '/' + folder_name + '/'

    if not os.path.exists(working_path):
        os.mkdir(working_path)  # create the individual folder

    # Copy required sim files to subdirectory.
    for file in REQUIRED_SIM_FILES:
        os.system(f'{copy_command} {HOME_PATH}{file} {working_path}')

    # Copy required *.cas *.dat files to subdirectory.
    s = 'TimeStep-' + str(t_start) + '*'
    os.system(f'{copy_command} {HOME_PATH}{s} {working_path}')

    # Create the text file which will tell sim at what timestep to restart at
    with open(working_path + T_START_NAME, 'w') as f:
        f.write(str(t_start))
        f.write("\n")
        f.write(str(t_end))

    # Create motor_parameters.txt file for the sim
    create_omega_txt(amp=motor_params['amp'],
                               freq=motor_params['freq'],
                               phase=motor_params['phase'],
                               offset=motor_params['offset'],
                               path=working_path,
                               t_end=t_end)

    if submit_job:
        os.chdir(working_path)
        os.system(f'sbatch {FLUENT_JOB_FILE}')
        os.chdir(RESULTS_FOLDER_PATH)


def check_if_individual_sim_finished(folder, t_end):
    # Checks to see if TimeStep-t_end.cas exists. If it does, sim == finished.
    s = folder + '/' + 'TimeStep-' + str(t_end) + '.cas'
    if os.path.exists(s):
        print(f'Sim in {folder} finished')
        return True
    else:
        print(f'Sim in {folder} NOT finished')
        return False


def check_if_iteration_sims_finished(folder, t_end):
    dirs = next(os.walk(folder))[1]
    temp = {dir: False for dir in dirs}
    while not all(elem for elem in temp.values()):
        print('-'*50)
        print(f'Waiting for sims in {folder} to finish')
        print('-'*50)
        for dir in dirs:
            s = folder + '/' + dir
            temp[dir] = check_if_individual_sim_finished(folder=s, t_end=t_end)
        time.sleep(SLEEP_TIME)
    print('-' * 50)
    print(f'Sims finished in {folder}')

    s = folder + '/' + dirs[0] + '/' + 'TimeStep-' + str(t_end) + '*'
    print(f'Copying {s} into home path.')
    os.system(f'{copy_command} {s} {HOME_PATH}')


def cleanup(folder):
    def _cleanup(path=os.getcwd(), clean=False):
        print('-'*50)
        print(f'Looking in {path}')
        filenames = next(os.walk(path))[-1]
        for filename in filenames:
            if all(s in filename for s in ['cleanup', 'fluent', '.sh']):
                print(f'Found {filename} in {path}')
                p = f'{path}/{filename}'
                if clean:
                    os.system(f'chmod +x {p}')
                    print('Running cleanup script...')
                    os.system(f'{p}')

    print(f'Running cleanup*.sh scripts')
    dirs = next(os.walk(folder))[1]
    for dir in dirs:
        s = folder + '/' + dir + '/'
        _cleanup(path=s, clean=True)


def compute_cost(sensor_data, motor_params):
    u_avg_top = np.mean(list(sensor_data['top'].values()))
    u_avg_mid = np.mean(list(sensor_data['mid'].values()))
    u_avg_bot = np.mean(list(sensor_data['bot'].values()))

    # Compute fluctuation cost
    sum = 0
    for t in sensor_data['top']:
        a = (sensor_data['top'][t] - u_avg_top)**2
        b = (sensor_data['mid'][t] - u_avg_mid)**2
        c = (sensor_data['bot'][t] - u_avg_bot)**2
        d = a + b + c
        sum += d

    n = len(sensor_data['top'])
    fluctuation_cost = (1 / (n*U_INF**2)) * sum

    # Compute actuation cost
    freq_front = motor_params['freq'][0]
    freq_top = motor_params['freq'][1]
    freq_bot = motor_params['freq'][2]

    freq_rms = math.sqrt(1/3 * (freq_front**2 + freq_top**2 + freq_bot**2))

    actuation_cost = (freq_rms * R) / U_INF

    total_cost = fluctuation_cost + GAMMA_COST_FACTOR*actuation_cost

    return total_cost


def CFD(iteration, motor_params, return_cost=True, return_sensor_data=False, run_CFD=True):
    t_start, t_end = get_t_start_end_at_n(iteration)
    # Returns CFD data from t_start+STEP to t_end (i.e. does not retrieve the data at timestep t_start)
    iteration_folder_path = f'{ITERATION_FOLDER_NAME}{str(iteration)}-{str(t_start)}-{str(t_end)}'

    if not os.path.exists(iteration_folder_path):
        os.mkdir(iteration_folder_path)

    print('-'*50)

    # Run CFD for each individual
    if run_CFD:
        for individual in motor_params:
            run_cfd_individual(path=iteration_folder_path,
                               motor_params=motor_params[individual],
                               t_start=t_start, t_end=t_end,
                               individual=individual,
                               submit_job=run_CFD)
        check_if_iteration_sims_finished(iteration_folder_path, t_end)

        # Execute cleanup*.sh scripts
        cleanup(iteration_folder_path)

    # Extract velocity data from iteration folder.
    data = {}
    for individual in motor_params:
        path = f'{iteration_folder_path}/{get_individual_folder_name(individual, t_start, t_end)}'
        data[individual] = data_functions.get_sensor_data(path=path, t_start=t_start, t_end=t_end, t_step=INTERVAL)

    # Compute the cost for each individual
    cost_data = {}
    for individual in motor_params:
        cost_data[individual] = compute_cost(data[individual], motor_params[individual])

    if return_cost:
        del data
        return cost_data
    elif return_sensor_data:
        del cost_data
        return data
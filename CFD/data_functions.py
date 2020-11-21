import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import multiprocessing as mp
import time
import os
import yaml

## Sensor Locations
chord = 0.02  # m
sensors = [
    (10*chord, 1.25*chord),   # top
    (10*chord, 0*chord),      # middle
    (10*chord, -1.25*chord),  # bot
]

sensor_data_yaml = 'sensor_data.yaml'
data_extraction_inputs = 'data_extraction_inputs.txt'
EXTRACT_VELOCITY_JOB_FILE = 'extract_velocity.slurm'

extract_data_sleep_time = 30  # s

def _parse_df(df):
    parsed_data = {}
    parsed_data['x-coords'] = []
    parsed_data['y-coords'] = []
    parsed_data['points'] = []

    for row in df.itertuples(index=False):

        x, y, p, x_vel, y_vel = row._1, row._2, row.pressure, row._4, row._5

        if x not in parsed_data:
            parsed_data[x] = {}

        parsed_data[x][y] = {'v': [x_vel, y_vel], 'p': p}

        parsed_data['x-coords'].append(x)
        parsed_data['y-coords'].append(y)
        parsed_data['points'].append([x, y])

    parsed_data['x-coords'] = np.sort(np.unique(parsed_data['x-coords']))
    parsed_data['y-coords'] = np.sort(np.unique(parsed_data['y-coords']))

    return parsed_data


def _get_val_in_df(x0, y0, df, var, d_max=5e-4):
    # Get the value at x, y coord.
    # var = ['p', 'v']

    # Set up KDTree
    tree = KDTree(df['points'])

    d, i = tree.query([x0, y0])  # tree.query returns (distance, index)
    if d > d_max:
        print(f'Warning d: {d} greater than specified d_max: {d_max}')
    x, y = tree.data[i]

    return df[x][y][var]


def _find_neighbors_in_list(x, l):
    # Finds neighbours of x in a list, returns x_1, x_2, where x_1 < x < x_2
    # l must be sorted smallest to largest
    if x in l:
        print('x is already in list')
        return [x]
    else:
        # Check if x is in bounds of list.
        if min(l) < x < max(l):
            temp = l.copy()
            temp = [elem for elem in temp if elem < x]
            lower_bound = temp[-1]

            temp = l.copy()
            temp = [elem for elem in temp if elem > x]
            upper_bound = temp[0]

            return [lower_bound, upper_bound]
        else:
            raise ValueError('x out of bounds.')


def extract_velocity_all(timesteps=None, t_start=None, t_end=None, t_step=None):
    # Old function which is not run in parallel.

    start_time = time.time()

    if timesteps is not None:
        files = timesteps
    elif t_start is not None and t_end is not None and t_step is not None:
        files = ['ascii' + str(i) for i in range(t_start, t_end + t_step, t_step)]
    else:
        raise ValueError('Invalid inputs to extract_velocity function.')

    sensor_values = {}
    sensor_values['top'] = {}
    sensor_values['mid'] = {}
    sensor_values['bot'] = {}

    for file in files:
        print(f'Extracting from {file}')
        # Get the timestep from the file name.
        timestep = int(file.split('ascii')[-1])

        # Load in raw ascii file into pandas dataframe.
        data = pd.read_csv(file, delim_whitespace=True, engine='c')

        # Parse dataframe according to how we want to structure the data.
        p_data = _parse_df(data)

        # Get the x-vel at sensor location using parsed data.
        for sensor_loc, sensorxy in zip(['top', 'mid', 'bot'], sensors):
            x, y = sensorxy[0], sensorxy[1]
            x_vel = _get_val_in_df(x, y, p_data, var='v')[0]
            sensor_values[sensor_loc][timestep] = x_vel

    print(f'Time taken to extract velocity data: {time.time() - start_time: 0.4f} s')

    return sensor_values


def extract_velocity(file, sensor_values):
    # Extract sensor x-velocity data from specified file.

    # sensor_values -- dict to store data

    print(f'Extracting from {file}')
    # Get the timestep from the file name.
    timestep = int(file.split('ascii')[-1])

    # Load in raw ascii file into pandas dataframe.
    data = pd.read_csv(file, delim_whitespace=True, engine='c')

    # Parse dataframe according to how we want to structure the data.
    p_data = _parse_df(data)

    # Get the x-vel at sensor location using parsed data.
    for sensor_loc, sensorxy in zip(['top', 'mid', 'bot'], sensors):
        x, y = sensorxy[0], sensorxy[1]
        x_vel = _get_val_in_df(x, y, p_data, var='v')[0]
        # multiprocessing can't handle nested dicts. therefore use the following code
        temp = sensor_values[sensor_loc]
        temp[timestep] = x_vel
        sensor_values[sensor_loc] = temp
        # below code would work if multiprocessing could handle nested dicts
        # sensor_values[sensor_loc][timestep] = x_vel

    print(f'Finished extracting {file}')


def mp_extract_velocity(path, t_start=None, t_end=None, t_step=None, timesteps=None,):
    print('-'*50)
    print(f'Extracting ascii data from {path}')

    # NEVER LOAD ASCII0
    if t_start == 0:
        t_start += t_step

    start_time = time.time()

    manager = mp.Manager()

    sensor_values = manager.dict()
    sensor_values['top'] = {}
    sensor_values['mid'] = {}
    sensor_values['bot'] = {}

    jobs = []

    if timesteps is not None:
        files = timesteps
    elif t_start is not None and t_end is not None and t_step is not None:
        files = ['ascii' + str(i) for i in range(t_start, t_end + t_step, t_step)]
    else:
        raise ValueError('Invalid inputs to extract_velocity function.')

    # Check if all files exist in folder:
    for file in files:
        if not os.path.exists(f'{path}/{file}'):
            print(f'{file} not found in {path}. Aborting...')
            exit()

    for file in files:
        working_path = path + '/' + str(file)
        p = mp.Process(target=extract_velocity, args=(working_path, sensor_values))
        jobs.append(p)
        p.start()

    for process in jobs:
        process.join()

    print(f'Time taken to extract velocity data: {time.time() - start_time: 0.4f} s')

    # Not sure why this occurs, but sometimes a single timestep will be missing from the dict.
    # This might be due to multiprocessing and running parallel.
    # Do a check to see if any missing timesteps. If missing, fill in as required.
    while True:
        print(f'Checking for missing values in velocity data')
        missing = []
        for timestep in range(t_start, t_end + t_step, t_step):
            for sensor_loc in ['top', 'mid', 'bot']:
                if timestep not in sensor_values[sensor_loc]:
                    print(f'Missing timestep {timestep} in {sensor_loc}. Filling in...')
                    filename = 'ascii' + str(timestep)
                    missing.append(filename)
        if not missing:
            break
        else:
            for file in missing:
                p = mp.Process(target=extract_velocity, args=(file, sensor_values))
                jobs.append(p)
                p.start()

            for process in jobs:
                process.join()

    return sensor_values._getvalue()


def get_sensor_data(path, t_start, t_end, t_step):
    # Output inputs into a txt file to be read in by extract_velocity.py script
    with open(f'../{data_extraction_inputs}', 'w') as f:
        f.write(path + '\n')
        f.write(str(t_start) + '\n')
        f.write(str(t_end) + '\n')
        f.write(str(t_step) + '\n')

    # Submit .slurm file calling "extract_velocity.py" script
    os.system(f'sbatch ../{EXTRACT_VELOCITY_JOB_FILE}')
    # main()

    # Wait until that job is finished (i.e. test if "sensor_data.yaml" exists)
    while not os.path.exists(f'{path}/{sensor_data_yaml}'):
        time.sleep(extract_data_sleep_time)

    # Read in "sensor_data.yaml" and reconstruct into sensor_data structure
    with open(f'{path}/{sensor_data_yaml}', 'r') as f:
        sensor_data = yaml.full_load(f)

    return sensor_data
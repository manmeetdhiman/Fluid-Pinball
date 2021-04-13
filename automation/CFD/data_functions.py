import numpy as np
import math
import pandas as pd
from scipy.spatial import KDTree
import multiprocessing as mp
import time
import os
import yaml
import json
import argparse
import settings


def output_yaml_json(filename, data):
    '''
    Outputs a variable into .json and .yaml files for storage.

    :param str filename:
    :param any data:
    :return: None
    '''

    # Filename must not include extension!
    with open(f'{filename}.yaml', 'w') as f:
        yaml.dump(data, f)
    with open(f'{filename}.json', 'w') as f:
        json.dump(data, f)


def _parse_df(df):
    '''
    Parses the FLUENT ascii data (i.e. dataframe (df)) into a readable format.

    :param df: Pandas dataframe.
    :return dict: A dictionary of the parsed data.
    '''

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


def _get_vel_data_in_df(sensors_locs, df, d_max=5e-4):
    # Get velocity data for all sensor locations in df.
    vel_data = {}
    tree = KDTree(df['points'])

    for location in sensors_locs:
        x0, y0 = sensors_locs[location]
        d, i = tree.query([x0, y0])
        if d > d_max:
            print(f'Warning d: {d} greater than specified d_max: {d_max}')
        x, y = tree.data[i]
        vel_data[location] = df[x][y]['v']

    return vel_data


def _extract_velocity(file):
    # Internal Worker function for the extract_velocity function
    # Extract x, y velocity data from single file.

    # sensor_values -- dict to store data
    print(f'Extracting from {file}')

    # Get the timestep from the file name.
    timestep = int(file.split('ascii')[-1])

    # Load in raw ascii file into pandas dataframe.
    data = pd.read_csv(file, delim_whitespace=True, engine='c')

    # Parse dataframe according to how we want to structure the data.
    p_data = _parse_df(data)

    # Get the x, y vel at sensor locations using parsed data.
    _data = _get_vel_data_in_df(settings.SENSOR_LOCATIONS, p_data)

    _temp = {}
    _temp[timestep] = _data

    print(f'Finished extracting {file}')

    return _temp


def extract_velocity(path):
    # Returns sensor velocity magnitude values for all "ascii" files in path.
    print('-' * 50)
    print(f'Extracting ascii data from {path}')

    start_time = time.time()

    sensor_data = {}
    sensor_data['top'] = {}
    sensor_data['mid'] = {}
    sensor_data['bot'] = {}

    all_files_folders = os.listdir(path=path)
    path_files = [file for file in all_files_folders if 'ascii' in file]

    with mp.Pool() as p:
        data = p.map(_extract_velocity, path_files)

    # Reconstruct master dataset from parallel data
    for dataset in data:
        timestep =  list(dataset.keys())[0]
        for location in dataset[timestep]:
            x_vel = dataset[timestep][location][0]
            y_vel = dataset[timestep][location][1]
            # IMPORTANT: Return magnitude of velocity at sensors.
            magnitude = math.sqrt(x_vel**2 + y_vel**2)
            sensor_data[location][timestep] = magnitude

    print(f'Time taken to extract velocity data: {time.time() - start_time: 0.4f} s')

    return sensor_data


def get_sensor_data(path):
    '''
    Gets the sensor data located in the given path.
    '''

    # If sensor data doesn't exist, output it.
    if not os.path.exists(f'{path}/{settings.SENSOR_DATA_YAML_FILE}'):
        print(f'Outputting sensor data in {os.getcwd()}/{path}')

        # Wait until "sensor_data.yaml" exists)
        while not os.path.exists(f'{path}/{settings.SENSOR_DATA_YAML_FILE}'):
            print(f'Sleeping for {settings.EXTRACT_DATA_SLEEP_TIME} s')
            time.sleep(settings.EXTRACT_DATA_SLEEP_TIME)
            print(f'Waiting for yaml sensor data in {os.getcwd()}/{path}')

    # Read in "sensor_data.yaml" and reconstruct into sensor_data structure
    with open(f'{path}/{settings.SENSOR_DATA_YAML_FILE}', 'r') as f:
        sensor_data = yaml.safe_load(f)

    return sensor_data


def main(path='.'):
    def output_sensor_data_as_yaml(path, data):
        with open(f'{path}/{settings.SENSOR_DATA_JSON_FILE}', 'w') as f:
            json.dump(data, f)
        with open(f'{path}/{settings.SENSOR_DATA_YAML_FILE}', 'w') as f:
            yaml.dump(data, f)

    data = extract_velocity(path)
    output_sensor_data_as_yaml(path=path, data=data)


if __name__ == '__main__':
    # Ignore this segment of code. It is NOT used by the automation. 
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = args.path
    main(path=path)

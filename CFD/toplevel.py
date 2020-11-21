import os
import cfd
from pprint import pprint

RESULTS_FOLDER_NAME = 'Results2'

# # Delete after
params = {'amp': [1, 1, 1], 'phase': [2, 2, 2], 'freq': [1, 2, 3], 'offset': [1, 3, 4]}
params2 = {'amp': [5, 5, 5], 'phase': [2, 2, 2], 'freq': [1, 2, 3], 'offset': [1, 3, 4]}


if __name__ == '__main__':
    if os.path.exists(RESULTS_FOLDER_NAME):
        os.chdir(RESULTS_FOLDER_NAME)
    else:
        os.mkdir(RESULTS_FOLDER_NAME)
        os.chdir(RESULTS_FOLDER_NAME)

    # Example: GA
    generations = 5

    # Example: RL
    # policies

    for generation in range(1, generations+1):
        # Get motor param data from GA
        motor_params = {1: params, 2: params2}

        # Run CFD and extract velocity data for all the individuals specified in motor_params dictionary
        sensor_data = cfd.CFD(generation, motor_params, run_CFD=False)

        # Call GA update
        pprint(sensor_data)

    print('-'*50)
    print('Finished all simulations.')




#!/usr/bin/env python3

import os
import sys

sys.path.append(f'./CFD')

from CFD import settings
from CFD import submit_workers

if __name__ == '__main__':
    if os.path.exists(settings.RESULTS_FOLDER_NAME):
        os.chdir(settings.RESULTS_FOLDER_NAME)
    else:
        os.mkdir(settings.RESULTS_FOLDER_NAME)
        os.chdir(settings.RESULTS_FOLDER_NAME)

    print(f'Current directory: {os.getcwd()}')

    submit_workers.main()

    print(f'Starting {settings.ALGORITHM_TYPE}')

    if settings.ALGORITHM_TYPE == 'RL':
        sys.path.append(f'{settings.HOME_PATH}/RL')
        from RL import RL_main as RL
        RL.main()
    elif settings.ALGORITHM_TYPE == 'GA':
        sys.path.append(f'{settings.HOME_PATH}/SOGA')
        from SOGA import main as GA
        GA.main()

    print('-'*50)
    print('Finished all simulations.')

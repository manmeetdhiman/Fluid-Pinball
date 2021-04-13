import os
import time
import numpy as np
import math
import yaml
from CFD import data_functions as data_functions
import settings
from pprint import pprint


def RL_create_motor_rotations_file(path, rotations, iteration, action, t_start, t_end):
    '''
    Creates the MOTOR_ROTATIONS_FILE needed for the RL CFD sims. Uses motor rotations computed by the RL code.

    :param str path: The path to output the motor rotations file.
    :param list rotations: The motor rotations in [ [front], [top], [bot] ] nested list format.
    :param int iteration: The iteration number of the RL.
    :param int action: The action number of the iteration.
    :param int t_start: The starting timestep of the action duration.
    :param int t_end: The ending timestep of the action duration.
    :return: None
    '''

    if action == 1:
        with open(f'{path}/{settings.MOTOR_ROTATIONS_FILE}', 'w') as f:
            rotations_list_counter = 0
            for timestep in range(t_start, t_end):
                if timestep == 0:
                    w_front = 0
                    w_top = 0
                    w_bot = 0
                else:
                    w_front = rotations[0][rotations_list_counter]
                    w_top = rotations[1][rotations_list_counter]
                    w_bot = rotations[2][rotations_list_counter]
                    rotations_list_counter += 1
                f.write(f'{timestep} {w_front: 0.5f} {w_top: 0.5f} {w_bot: 0.5f}\n')
    else:
        # Copy previous action's motor_parameters.txt into cwd and append.
        for foldername in os.listdir(f'{settings.ITERATION_FOLDER_NAME}-{iteration}'):
            if f'action-{action - 1}-' in foldername:
                previous_action_folder_name = foldername
                break
        # previous_action_folder_name = get_individual_folder_name(action - 1,
        #                                                          t_start=t_start - (t_end - t_start),
        #                                                          t_end=t_start)
        previous_action_path = f'{settings.ITERATION_FOLDER_NAME}-{iteration}/{previous_action_folder_name}'
        os.system(f'{settings.COPY_COMMAND} {previous_action_path}/{settings.MOTOR_ROTATIONS_FILE} {path}')
        # MOTOR_ROTATIONS_FILE needs to be edited to include the t_start-1 timestep and its data.
        previous_action_data = np.loadtxt(f'{previous_action_path}/{settings.MOTOR_ROTATIONS_FILE}')
        previous_action_last_timestep_data = previous_action_data[-1]
        w_front_last_action = float(previous_action_last_timestep_data[1])
        w_top_last_action = float(previous_action_last_timestep_data[2])
        w_bot_last_action = float(previous_action_last_timestep_data[3])
        with open(f'{path}/{settings.MOTOR_ROTATIONS_FILE}', 'a') as f:
            rotations_list_counter = 0
            for timestep in range(t_start, t_end):
                if timestep == t_start:
                    w_front = w_front_last_action
                    w_top = w_top_last_action
                    w_bot = w_bot_last_action
                else:
                    w_front = rotations[0][rotations_list_counter]
                    w_top = rotations[1][rotations_list_counter]
                    w_bot = rotations[2][rotations_list_counter]
                    rotations_list_counter += 1
                f.write(f'{timestep} {w_front: 0.5f} {w_top: 0.5f} {w_bot: 0.5f}\n')


def GA_create_motor_rotations_file(path, rotations):
    """
    Creates the MOTOR_ROTATIONS_FILE needed for the GA CFD sims. Uses motor rotations computed by the GA code.

    :param str path: The path to output the MOTOR_ROTATIONS_FILE.
    :param list rotations: The rotation data in [ [front], [top], [bot] ] format.
    :return: None
    """
    with open(f'{path}/{settings.MOTOR_ROTATIONS_FILE}', 'w') as f:
        for timestep in range(len(rotations[0])):
            w_front = rotations[0][timestep]
            w_top = rotations[1][timestep]
            w_bot = rotations[2][timestep]
            f.write(f'{timestep + 1} {w_front: 0.5f} {w_top: 0.5f} {w_bot: 0.5f}\n')


def create_perturbation_inlet_velocity(path, amp, freq, phase, offset, base,
                                       t_start, t_end,
                                       total=10000, n_periods=1, spacing=20, dt=5e-4):
    '''
    Creates the turbulent with perturbation inlet_velocity text file for CFD simulation.
    The perturbation is represented by mathematically by P = amp*sin(freq*t + offset)

    :param str path: The path to output inlet_velocity file.
    :param int amp: The amplitude of the perturbation (e.g. base 1.5 m/s + perturbation amplitude of 0.1 m/s).
    :param int freq: The frequency of the perturbation.
    :param int phase: The phase of the perturbation.
    :param int offset: The offset of the perturbation.
    :param int base: The base velocity without perturbation.
    :param int t_start: The timestep where perturbation starts.
    :param int t_end: The timestep where perturbation ends.
    :param int total: The total number of timesteps to generate. If total > t_end, then inlet velocity set to base.
    :param int n_periods: The number of periods for each perturbation (e.g. if n_periods=2, then each perturbation lasts 2 sin waves).
    :param int spacing: The number of timesteps between each sin wave perturbation.
    :param dt: The CFD timestep size.
    :return: None
    '''



    # Assumes phase in RADIANS.
    v = lambda t: amp*math.sin(freq*t + phase) + offset
    t_period = ( 2*math.pi - phase ) / (freq)
    n = math.floor(t_period * n_periods / dt)  # timesteps needed for one period
    # print(f'Timesteps needed for one period {n}')

    with open(f'{path}/{settings.INLET_VELOCITY_FILE}', 'w') as f:
        for timestep in range(1, t_start):
            f.write(f'{timestep} {base} \n')
        counter = 0
        spacing_counter = 0
        for timestep in range(t_start, t_end + 1):
            if 0 <= counter <= n:
                v1 = v(counter * dt) + base
                if v1 < 0:
                    exit('Inlet velocity is less than zero')
                f.write(f'{timestep} {v1} \n')
                counter += 1
            elif counter > n and spacing_counter < spacing:
                f.write(f'{timestep} {base} \n')
                spacing_counter += 1
            elif spacing_counter >= spacing:
                f.write(f'{timestep} {base} \n')
                counter = 0
                spacing_counter = 0
        for timestep in range(t_end + 1, total + 1):
            f.write(f'{timestep} {base} \n')


def get_individual_folder_name(individual, t_start, t_end):
    '''
    Gets the folder name of a single CFD individual sim.

    :param int individual: A number/tag representing the individual's ID.
    :param int t_start: The starting timestep of the CFD simulation for this individual.
    :param int t_end: The ending timestep of the CFD simulation for this individual.
    :return str: The folder name (e.g. individual-9-0-2000).
    '''
    return f'{settings.INDIVIDUAL_FOLDER_NAME}-{individual}-{t_start}-{t_end}'


def check_if_sim_finished(path, t_end):
    '''
    # Checks to see if TimeStep-t_end.cas exists. If it does, sim == finished.

    :param str path: The folder path where you want to check if the CFD simulation is finished.
    :param int t_end: The ending timestep of the CFD simulation.
    :return bool:
    '''

    s = f'{path}/TimeStep-{t_end}.cas'
    if os.path.exists(s):
        print(f'Sim in {os.getcwd()}/{path} finished')
        return True
    else:
        print(f'Sim in {os.getcwd()}/{path} NOT finished')
        return False


def remove_ascii(path):
    '''
    Removes ascii files in the given path to save space.

    :param str path: The folder path where ascii files need to be removed.
    :return: None
    '''

    print(f'Removing ascii* in {os.getcwd()}/{path}')
    os.system(f'rm {os.getcwd()}/{path}/ascii*')


def remove_cas_dat(path):
    '''
    Removes *cas/dat files in the given path to save space.

    :param str path: The folder path where *cas/dat files files need to be removed.
    :return: None
    '''

    print(f'Removing *.cas *.dat in {os.getcwd()}/{path}')
    os.system(f'rm {os.getcwd()}/{path}/*.dat')
    os.system(f'rm {os.getcwd()}/{path}/*.cas')


def compute_cost(sensor_data, motor_params):
    '''
    Compute the cost of the CFD simulation based on the simulation's sensor signals.

    :param dict sensor_data: {'top': [...], 'mid': [...], 'bot': [...]} where the lists are the sensor signals.
    :param dict motor_params: The dictionary of motor_params. The format is in the comment below.
        motor_params structure:
        motor_params = {
                        'amp': [1, 2, 3], # front, top, bot motor order
                        'freq': [1, 2, 3],
                        'phase': [1, 2, 3],
                        'offset': [1, 2, 3]
                        'revolutions': [ [front], [top], [bot] ]
                        }
    :return dict: Dictionary of fluctuation cost, actuation cost, and total cost.
    '''

    # Check if 'revolutions' key is in motor_params.
    if 'revolutions' not in motor_params:
        raise KeyError(f'Revolutions key not in motor_params dictionary. Exiting...')

    # Compute fluctuation cost
    u_avg_top = np.mean(list(sensor_data['top'].values()))
    u_avg_mid = np.mean(list(sensor_data['mid'].values()))
    u_avg_bot = np.mean(list(sensor_data['bot'].values()))

    timesteps = sensor_data['top'].keys()
    sum = 0
    for timestep in timesteps:
        a = (sensor_data['top'][timestep] - u_avg_top)**2
        b = (sensor_data['mid'][timestep] - u_avg_mid)**2
        c = (sensor_data['bot'][timestep] - u_avg_bot)**2
        d = a + b + c
        sum += d
    n = len(sensor_data['top'])  # doesn't matter, len top, mid, bot should all be the same
    fluctuation_cost = (1 / (3*n*settings.U_INF**2)) * sum

    # Compute actuation cost
    sum = 0
    for theta_front, theta_top, theta_bot in zip(motor_params['revolutions'][0],
                                                 motor_params['revolutions'][1],
                                                 motor_params['revolutions'][2]):
        d = theta_front**2 + theta_top**2 + theta_bot**2
        sum += d
    n = len(motor_params['revolutions'][0])  # doesn't matter, len front, top, bot should all be the same
    actuation_cost = settings.R / settings.U_INF * math.sqrt( (1/(3*n)) * sum)

    # total_cost = fluctuation_cost + settings.GAMMA_COST_FACTOR*actuation_cost

    # New cost filter function
    print(f'Actuation cost: {actuation_cost}, Fluctation cost: {fluctuation_cost}')
    total_cost = math.tanh(0.7*actuation_cost) + math.tanh(12.2*fluctuation_cost)

    cost_data = {}
    cost_data['Jfluc'] = float(fluctuation_cost)
    cost_data['Jact'] = float(actuation_cost)
    cost_data['Jtotal'] = float(total_cost)

    return cost_data


def _run_CFD_sim(t_start, t_end, working_path, motor_params, cas_dat_file_path):
    '''
    Internal function which creates the folder for CFD sim, and copies the necessary input files.

    :param int t_start: Starting CFD timestep.
    :param int t_end: Ending CFD timestep.
    :param str working_path: The path where the CFD folder should be created.
    :param dict motor_params: The motor_params dictionary. (See compute_cost function for format).
    :param str cas_dat_file_path: The folder path where the initial input files are stored. (e.g. TimeStep-0.dat).
    :return:
    '''

    if not os.path.exists(working_path):
        print(f'Creating CFD folder: {os.getcwd()}/{working_path}')
        os.makedirs(working_path)

        # Copy required sim files to subdirectory.
    for file in settings.REQUIRED_SIM_FILES:
        os.system(f'{settings.COPY_COMMAND} {settings.INIT_FILE_PATH}/{file} {working_path}')

        # Copy required *.cas *.dat files to subdirectory.
    s = f'TimeStep-{str(t_start)}*'
    os.system(f'{settings.COPY_COMMAND} {cas_dat_file_path}/{s} {working_path}')

    # Create the text file which will tell sim at what timestep to restart at
    with open(f'{working_path}/{settings.T_START_NAME}', 'w') as f:
        f.write(str(t_start))
        f.write("\n")
        f.write(str(t_end))

    # Output motor params to yaml file. First, reorganize motor params data into a dict.
    motor_params_data = {}
    for i, loc in enumerate(['front', 'top', 'bot']):
        motor_params_data[loc] = {}
        for param in ['amp', 'freq', 'phase', 'offset']:
            motor_params_data[loc][param] = motor_params[param][i]

    data_functions.output_yaml_json(filename=f'{working_path}/{settings.MOTOR_PARAMS_FILENAME}',
                                    data=motor_params_data)

    create_perturbation_inlet_velocity(path=working_path, amp=settings.U_INF * 0.1, freq=900,
                                       phase=0, offset=0, base=settings.U_INF,
                                       t_start=0, t_end=10000, spacing=20, total=10000)


def _GA_run_CFD_sim(t_start, t_end, working_path, motor_params,
                   cas_dat_file_path):
    '''
    Creates the CFD folder for a GA's individual. Input parameters are identical to _run_CFD_sim.
    '''

    _run_CFD_sim(t_start=t_start, t_end=t_end, working_path=working_path,
                 motor_params=motor_params, cas_dat_file_path=cas_dat_file_path)
    # Create motor_parameters.txt file for the sim
    GA_create_motor_rotations_file(path=working_path, rotations=motor_params['revolutions'])


def _RL_run_CFD_sim(t_start, t_end, working_path, iteration, action, motor_params,
                   cas_dat_file_path):
    '''
    Creates the CFD folder for an RL's individual. Input parameters are identical to _run_CFD_sim.

    :param int iteration: The iteration number.
    :param int action: The action number of the associated iteration.
    '''

    _run_CFD_sim(t_start=t_start, t_end=t_end, working_path=working_path,
                 motor_params=motor_params, cas_dat_file_path=cas_dat_file_path)
    # Create motor_parameters.txt file for the sim
    RL_create_motor_rotations_file(path=working_path, rotations=motor_params['revolutions'],
                                   iteration=iteration, action=action,
                                   t_start=t_start, t_end=t_end)


def GA_run_CFD(generation, population, skip_transient=True):
    '''
    Main run CFD function that is called by the GA code.

    :param int generation: The generation number.
    :param dict population: The population of individuals that need to be CFD simulated.
        population dictionary format:
        population = {individual1_object, individual2_object, ...} (See GA code for for individual object class).
    :param bool skip_transient: Skip the first 1000 timesteps in the cost calculation to avoid the transient region.
    :return dict: A dictionary of the cost and sensor signals for each individual given in the population dictionary.
    '''

    print('-'*50)
    print(f'Running CFD for Generation {generation}')
    print('-'*50)

    # Parse GA data into format for CFD
    def _parse_GA_for_CFD(population):
        # population is a list of Individual objects
        motor_params = {}
        for individual in population:
            motor_params[individual.id] = individual.genes.copy()
            motor_params[individual.id]['freq'] = motor_params[individual.id].pop('frequency')
            motor_params[individual.id]['amp'] = motor_params[individual.id].pop('amplitude')
            motor_params[individual.id]['revolutions'] = individual.revolutions.copy()

        return motor_params

    def _get_t_start_end_at_n(n):
        # Iteration must start at 0
        t_start = n * settings.TOTAL_TIMESTEPS
        t_end = t_start + settings.TOTAL_TIMESTEPS
        return t_start, t_end

    motor_params = _parse_GA_for_CFD(population=population)

    # Start the next iteration from timestep 0 again
    t_start, t_end = _get_t_start_end_at_n(0)

    print('-'*50)

    individuals_to_sim = list(motor_params.keys())

    print(f'Individuals to sim: {individuals_to_sim}')
    for individual in individuals_to_sim:
        individual_folder_path = get_individual_folder_name(individual, t_start, t_end)
        _GA_run_CFD_sim(t_start=t_start, t_end=t_end,
                        working_path=individual_folder_path,
                        motor_params=motor_params[individual],
                        cas_dat_file_path=settings.INIT_FILE_PATH)
        job_name = f'{settings.RESULTS_FOLDER_PATH}/{settings.WORKER_JOB_FILE}-{individual}.txt'
        print(f'Outputting individual-{individual} job')
        with open(job_name, 'w') as f:
            f.write(f'{os.getcwd()}/{individual_folder_path}')
            f.write(f'\n')

    # Check if sims are finished
    individuals_to_check = list(motor_params.keys())
    temp = individuals_to_check # this list reduces to empty list when all individuals done
    while individuals_to_check != []:
        print('-'*50)
        print(f'Current Generation: {generation}')
        for individual in individuals_to_check:
            individual_folder_path = get_individual_folder_name(individual, t_start, t_end)
            if check_if_sim_finished(path=individual_folder_path, t_end=t_end):
                # Output the sensor data into file
                data_functions.get_sensor_data(path=individual_folder_path)
                temp.remove(individual)  # no longer need to check this individual anymore
                individual_folder_path = get_individual_folder_name(individual, t_start, t_end)
                remove_ascii(path=individual_folder_path)
                remove_cas_dat(path=individual_folder_path)


        individuals_to_check = temp
        print('-'*50)
        print(f'Sleeping for {settings.SLEEP_TIME}s')
        time.sleep(settings.SLEEP_TIME)

    # Remove ascii* *dat *cas files
    # for individual in motor_params:
    #     individual_folder_path = get_individual_folder_name(individual, t_start, t_end)
    #     remove_ascii(path=individual_folder_path)
    #     remove_cas_dat(path=individual_folder_path)

    # Load in sensor data
    sensor_data = {}
    for individual in motor_params:
        individual_folder_path = f'{get_individual_folder_name(individual, t_start, t_end)}'

        # Read in "sensor_data.yaml" and reconstruct into sensor_data structure
        with open(f'{individual_folder_path}/{settings.SENSOR_DATA_YAML_FILE}', 'r') as f:
            data = yaml.safe_load(f)

        if skip_transient:
            # Only return last 1k timesteps to avoid transient region.
            _temp = {}
            for loc in data:
                _temp[loc] = {}
                for timestep in data[loc]:
                    if timestep >= 1000:
                        _temp[loc][timestep] = data[loc][timestep]

            data = _temp

        sensor_data[individual] = data

    # Compute the cost for each individual
    cost_data = {}
    for individual in motor_params:
        cost_data[individual] = compute_cost(sensor_data[individual], motor_params[individual])

    master_data = cost_data.copy()
    for individual in motor_params:
        master_data[individual]['sensor_data'] = sensor_data[individual]

    return master_data


def RL_run_CFD(iteration, action, t_start, t_end, motor_params):
    '''
    Main run CFD function that is called by the RL code.

    :param int iteration: The generation number.
    :param int t_start: The starting CFD timestep of the action.
    :param int t_end: The ending CFD timestep of the action.
    :param dict motor_params: The dictionary of motor_params.
    :return dict: A dictionary of the sensor signals that the RL code uses to compute the cost, etc.
    '''

    t_start -= 1

    # Get path of previous iteration so we can copy the *.dat and *.cas files for simulation.
    if action == 1:  # first iteration, copy from home path
        init_file_path = settings.INIT_FILE_PATH
    else:
        for foldername in os.listdir(f'{settings.ITERATION_FOLDER_NAME}-{iteration}'):
            if f'action-{action-1}-' in foldername:
                previous_action_folder_name = foldername
                break
        # previous_action_folder_name = get_individual_folder_name(action - 1,
        #                                                          t_start=t_start - (t_end - t_start),
        #                                                          t_end=t_start)
        previous_action_path = f'{settings.ITERATION_FOLDER_NAME}-{iteration}/{previous_action_folder_name}'
        init_file_path = previous_action_path

    action_folder_name = get_individual_folder_name(action, t_start, t_end)
    working_path = f'{settings.ITERATION_FOLDER_NAME}-{iteration}/{action_folder_name}'

    _RL_run_CFD_sim(t_start=t_start, t_end=t_end,
                   working_path=working_path,
                   iteration=iteration, action=action,
                   motor_params=motor_params,
                   cas_dat_file_path=init_file_path)

    job_name = f'{settings.RESULTS_FOLDER_PATH}/{settings.WORKER_JOB_FILE}-{iteration}-{action}.txt'

    print(f'Outputting iteration-{iteration} action-{action} job')
    with open(job_name, 'w') as f:
        f.write(f'{os.getcwd()}/{working_path}')
        f.write(f'\n')

    while not check_if_sim_finished(path=working_path, t_end=t_end):
        print('-' * 50)
        print(f'Sleeping for {settings.SLEEP_TIME}s')
        time.sleep(settings.SLEEP_TIME)

    # Output the sensor data into file
    data_functions.get_sensor_data(path=working_path)

    # Safe check to ensure that sensor_data is not None
    # This is a manual hotfix to errors that occur on the clusters
    sensor_data = None
    while sensor_data is None:
        print(f'Loading sensor data from {working_path}')
        with open(f'{working_path}/{settings.SENSOR_DATA_YAML_FILE}') as f:
            sensor_data = yaml.safe_load(f)
        time.sleep(5)

    # Slightly modify how sensor_data is structured for RL purposes
    for loc in sensor_data:
        # Instead of sensor_data['top'] = {5: 1.5, 10: 1.75}, use list [1.5, 1.75].
        # Timestep is implied in the index of the list.
        sensor_data[loc] = list(sensor_data[loc].values())

    remove_ascii(path=working_path)

    if action != 1:
        # Remove the *.cas *.dat of the previous action since we are done with it.
        # init_file_path == previous_action_path
        remove_cas_dat(path=init_file_path)

    return sensor_data


def RL_get_sensor_data(iteration, action, t_start, t_end):
    '''
    # This is a manual hotfix to the issues that arise on the cluster. Gets the sensor data of the RL iteration-action.

    :param int iteration: The generation number.
    :param int action: The action number.
    :param int t_start: The starting CFD timestep of the action.
    :param int t_end: The ending CFD timestep of the action.
    :return dict: A dictionary of the sensor signals.
    '''

    action_folder_name = get_individual_folder_name(action, t_start, t_end)
    working_path = f'{settings.ITERATION_FOLDER_NAME}-{iteration}/{action_folder_name}'
    sensor_data = None
    while sensor_data is None:
        print(f'Loading sensor data from {working_path}')
        with open(f'{working_path}/{settings.SENSOR_DATA_YAML_FILE}') as f:
            sensor_data = yaml.safe_load(f)
        time.sleep(5)

    for loc in sensor_data:
        # Instead of sensor_data['top'] = {5: 1.5, 10: 1.75}, use list [1.5, 1.75].
        # Timestep is implied in the index of the list.
        sensor_data[loc] = list(sensor_data[loc].values())

    return sensor_data
# OLD deprecated function
# def create_folders_run_cfd(t_start, t_end, individuals, submit_job=False):
#     print(f'Creating CFD folders for t_start: {t_start}')
#
#     # Create iteration master folder
#     timesteps = str(t_start) + '-' + str(t_end)  # e.g. 0-10
#     master_folder_name = ITERATION_FOLDER_NAME + timesteps  # e.g. iteration-0-10
#     os.mkdir(master_folder_name)
#     os.chdir(master_folder_name)
#
#     # Create individual folders in iteration master folder
#     for _ in range(len(individuals)):
#         sub_folder_name = INDIVIDUAL_FOLDER_NAME + str(_) + '-' + timesteps
#         os.mkdir(sub_folder_name)
#         os.chdir(sub_folder_name)
#
#         # Copy required sim files to subdirectory.
#         for file in REQUIRED_SIM_FILES:
#             os.system(f'{copy_command} {HOME_PATH}{file} .')
#
#         # Copy required *.cas *.dat files to subdirectory.
#         s = 'TimeStep-' + str(t_start) + '*'
#         os.system(f'{copy_command} {HOME_PATH}{s} .')
#
#         # Create random motor parameters file. DELETE AFTER
#         with open('motor_parameters.txt', 'w') as f:
#             f.write(f'{random.randint(0, 20)} {random.randint(0, 20)} {random.randint(0, 20)}')
#
#         # Create the text file which will tell sim at what timestep to restart at
#         with open(T_START_NAME, 'w') as f:
#             f.write(str(t_start))
#             f.write("\n")
#             f.write(str(t_end))
#
#         if submit_job:
#             os.system(f'sbatch {FLUENT_JOB_FILE}')
#
#         os.chdir('../')  # go back into master iteration folder
#
#     os.chdir('../') # go back into Results2 folder

# OLD Toplevel main
#  for i, t in enumerate(range(START_TIME_STEP, END_TIME_STEP, INTERVAL)):  # for iteration in iterations. # governed by ML tool.
#         t_start
# #         t_end = t + INTERVAL
# #         # timesteps = str(t_sta= trt) + '-' + str(t_end)  # e.g. 0-10
#         iteration_folder_path = ITERATION_FOLDER_NAME + f'{i+1}-{str(t_start)}-{str(t_end)}'  # e.g. iteration-0-10
#         # os.mkdir(iteration_folder_path)
#
#         print('-'*50)
#
#         # CFD
#         # for individual in individuals:
#         #     run_cfd_individual(path=iteration_folder_path, motor_params=params, t_start=t, t_end=t_end,
#         #                        individual=individual,
#         #                        submit_job=False)
#         check_if_iteration_sims_finished(iteration_folder_path, t_end)
#
#         # Execute cleanup*.sh scripts
#         # cleanup(iteration_folder_path)
#
#         # Extract velocity data from iteration folder.
#         for individual in individuals:
#             path = f'{iteration_folder_path}/{get_individual_folder_name(individual, t_start, t_end)}'
#             data[individual] = pinball.mp_extract_velocity(path=path, t_start=5, t_end=20, t_step=INTERVAL)
#
#         pprint(data)
#
#         # Call ML algorithm, feed in extracted velocity data.

# if __name__ == '__main__':
    # create_omega_txt(path='./', amp=[0, -820, 820], freq=[50, 50, 50], phase=[0, 0, 0], offset=[0, 0, 0], t_end=10000)

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

if __name__ == '__main__':
    create_inlet_velocity(amp=3, freq=250, phase=0, offset=0, base=1.5, t_start=50, n_periods=0.5, t_end=200)


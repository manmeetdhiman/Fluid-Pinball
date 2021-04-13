def create_motor_rotations_file(amp, freq, phase, offset, t_end, path, dt=5e-4):
    # IMPORTANT

    # If you give a sim t_start = 0 and t_end = 4.
    # Output.txt gives t = 1, 2, 3, 4 data. NO data for t = 0.
    # Therefore, t-omega txt file should only have data for 1 2 3 4. NOT 0 1 2 3 4.
    # Otherwise, sim will use the first 4 rows (e.g. 0 1 2 3) instead of the correct (1 2 3 4)
    # Freq - Hz, Phase - rad, Offset/Amp - rad/s
    motor_speeds = [
        lambda t: amp[0]*math.sin(freq[0]*t + phase[0]) + offset[0],
        lambda t: amp[1]*math.sin(freq[1]*t + phase[1]) + offset[1],
        lambda t: amp[2]*math.sin(freq[2]*t + phase[2]) + offset[2]
    ]
    with open(f'{path}/{settings.MOTOR_ROTATIONS_FILE}', 'w') as f:
        for timestep in range(1, t_end + 1):
            t = timestep*dt
            w_front = motor_speeds[0](t)
            w_top = motor_speeds[1](t)
            w_bot = motor_speeds[2](t)
            f.write(f'{timestep} {w_front: 0.5f} {w_top: 0.5f} {w_bot: 0.5f}\n')


def edit_slurm_file_job_name(path, slurm_file, new_name):
    '''
    Opens a specified.slurm file and edits the job name entry.

    :param str path: The path where the *.slurm file is located.
    :param str slurm_file: The name of the slurm file (e.g. worker.slurm or fluent.slurm).
    :param str new_name: The new job name.
    :return: None
    '''

    with open(f'{path}/{slurm_file}', 'r') as f:
        new_fluent_slurm_lines = f.readlines()

    new_fluent_slurm_lines[2] = f'#SBATCH --job-name={new_name}\n'

    with open(f'{path}/{slurm_file}', 'w') as f:
        f.writelines(new_fluent_slurm_lines)
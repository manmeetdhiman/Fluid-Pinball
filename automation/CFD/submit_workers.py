import os
import settings

def main():
    worker_slurm = settings.WORKER_SLURM_FILE
    tag = settings.ALGORITHM_TYPE
    n_workers = settings.N_WORKERS
    path = settings.INIT_FILE_PATH
    ## -job-name must be on line 3 of the worker_slurm file!
    ## --output must be on line 4 of the worker_slurm file!

    print(f'Submitting worker jobs')

    for worker_id in range(n_workers):
        print(f'Submitting worker{worker_id} job')

        with open(f'{path}/{worker_slurm}', 'r') as f:
            new_fluent_slurm_lines = f.readlines()

        new_fluent_slurm_lines[2] = f'#SBATCH --job-name={tag}-w{worker_id}\n'
        new_fluent_slurm_lines[3] = f'#SBATCH --output={tag}-worker{worker_id}.out\n'

        with open(f'{path}/{worker_slurm}{worker_id}', 'w') as f:
            f.writelines(new_fluent_slurm_lines)

        os.system(f'dos2unix {path}/{worker_slurm}{worker_id}')

        os.system(f'sbatch {path}/{worker_slurm}{worker_id} {worker_id}')

if __name__ == '__main__':
   main()
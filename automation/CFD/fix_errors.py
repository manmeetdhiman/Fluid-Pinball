import time
import settings
import os
import subprocess

# Some CFD sims on the cluster may crash due to strange issues. This script addresses each of those issues manually.
# When a sim crashes, this script determines which simulation crashed and restarts the compute cores that were working on that simulation.

def main():
    worker_slurm = settings.WORKER_SLURM_FILE
    init_path = settings.INIT_FILE_PATH
    os.chdir(settings.RESULTS_FOLDER_PATH)
    print(f'CWD: {os.getcwd()}')

    def fix_cortex_error():
        cortex_paths = subprocess.Popen(f'find {settings.RESULTS_FOLDER_PATH} -name cortexerror.log',
                                        shell=True,
                                        stdout=subprocess.PIPE).stdout.read().decode('utf-8')
        cortex_paths = cortex_paths.split('\n')
        cortex_paths.remove('')

        for n, path in enumerate(cortex_paths):
            path = path.split('/cortexerror.log')[0]
            print(f'Found cortexerror.log in {path}')

            for filename in os.listdir(f'{path}'):
                if "worker" in filename:
                    with open(f'{path}/{filename}', 'r') as f:
                        worker_id = f.readline().strip()
                    worker_slurm_id = filename.split('_')[-1]
                    print(f'Worker: {worker_id}, Job_ID: {worker_slurm_id}')

                    os.system(f'sbatch {init_path}/{worker_slurm}{worker_id} {worker_id}')
                    os.system(f'scancel {worker_slurm_id}')
                    os.system(f'rm -r {path}/cortexerror.log')
                    os.system(f'rm -r {path}/{filename}')
                    if os.path.exists(f'{path}/output.txt'):
                        os.system(f'mv {path}/output.txt {path}/output2.txt')
                    if not os.path.exists(f'{path}/sensor_data.yaml'):
                        with open(f'{settings.WORKER_JOB_FILE}-cortex-{n}.txt', 'w') as f:
                            f.write(path + '\n')
                    break


    def fix_cluster_errors():
        keywords = ['Failed to connect', 'Permission denied', 'Connection closed by',
                    'Received signal SIGTERM']

        for keyword in keywords:
            command = f'grep --include=\*.out -rnwl \'{settings.RESULTS_FOLDER_PATH}\' -e "{keyword}"'
            paths = subprocess.Popen(command, shell=True,
                                 stdout=subprocess.PIPE).stdout.read().decode('utf-8')
            paths = paths.split('\n')
            paths.remove('')

            for n, path in enumerate(paths):
                path = path.split('/pinball.out')[0]
                print(f'{keyword} error in {path}')

                for filename in os.listdir(f'{path}'):
                    if "worker" in filename:
                        with open(f'{path}/{filename}', 'r') as f:
                            worker_id = f.readline().strip()
                        worker_slurm_id = filename.split('_')[-1]
                        print(f'Worker: {worker_id}, Job_ID: {worker_slurm_id}')

                        os.system(f'sbatch {init_path}/{worker_slurm}{worker_id} {worker_id}')
                        os.system(f'scancel {worker_slurm_id}')
                        os.system(f'rm -r {path}/pinball.out')
                        os.system(f'rm -r {path}/{filename}')
                        if os.path.exists(f'{path}/output.txt'):
                            os.system(f'mv {path}/output.txt {path}/output2.txt')
                        if not os.path.exists(f'{path}/sensor_data.yaml'):
                            with open(f'{settings.WORKER_JOB_FILE}-cluster-{n}.txt', 'w') as f:
                                f.write(path + '\n')
                        break

    while True:

        fix_cortex_error()

        fix_cluster_errors()

        time.sleep(30)


if __name__ == '__main__':
    main()



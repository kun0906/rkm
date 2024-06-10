"""
Run experiment on HPC

    ssh ky8517@tiger.princeton.edu

    cd /scratch/gpfs/ky8517/rkm/rkm_final
    module purge
    module load anaconda3/2021.11
    #conda env list
    #conda create --name py3104_rkm python=3.10.4
    #conda activate py3104
    #pip install -r ../requirements.txt  # you should install in the login node (not in compute nodes)

    pwd
    python3 -V
    uname -a

    python3 hpc_sbatch.py


    # download the remote results to local
    rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/rkm/rkm_final/out .

    squeue --format="%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R" --me
"""

import os.path
import shutil
import subprocess

import argparse
import os
import subprocess
from tqdm import tqdm
from functools import partial
import numpy as np
print = partial(print, flush=True)

parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--n_repeats", type=int, default=1001)  #
args = parser.parse_args()
print(args)


# project_dir = '~/'
# os.chdir(project_dir)

def check_dir(in_dir):
    # if os.path.exists(in_dir):
    #     shutil.rmtree(in_dir)

    # if os.path.isfile(in_pth):
    #     # To find whether a given path is an existing regular file or not.
    #     in_dir = os.path.dirname(in_pth)
    # elif os.path.isdir(in_pth):
    #     in_dir = in_pth
    # else:
    #     raise ValueError(in_pth)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

    return


def generate_sh(name, cmd, log_file):
    out_dir = os.path.dirname(log_file)
    mm = np.random.randint(10, high=60, size=1)[0]  # [`low`, `high`)
    s = fr"""#!/bin/bash
#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=24:{mm}:00          # total run time limit (HH:MM:SS)
#SBATCH --output={out_dir}/out.txt
#SBATCH --error={out_dir}/err.txt

date
module purge
cd /scratch/gpfs/ky8517/rkm/rkm_final
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
#conda activate py3104
#pip install -r ../requirements.txt 

pwd
python3 -V
uname -a 


{cmd} &> {log_file} 

# python3 main_clustering_diffdim_random.py --n_repeats n_repeats --true_cluster_size true_cluster_size --init_method init_method &> 'out/main_clustering_diffdim_{name}.txt' & 
# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
date
echo 'done'     
"""
    out_sh = f'{out_dir}/sbatch.sh'
    # print(out_sh)
    with open(out_sh, 'w') as f:
        f.write(s)

    cmd = f'sbatch {out_sh}'
    ret = subprocess.run(cmd, shell=True)
    print(cmd, ret)


def main():
    OUT_DIR = "out"
    cnt = 0
    procs = set()
    # for synthetic datasets
    for n_repeats in [args.n_repeats]:  # [5000]
        for true_cluster_size in [100]:
            for std in [2]:  # [0.5, 1, 2]: #[0.1, 0.25, 0.5, 1, 0.1, 0.25, ]:
                for n_neighbours in [0]:  # nn = [10, 25, 50, 100, 200], nn = [5, 10, 15, 20, 25, 50],  theta = [10, 50, 100, 250, 500], m = [0.1, 0.2, 0.3, 0.4, 0.5], projected_dimensional = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    for with_outlier in [True]:  # [True, False]:
                        for init_method in ['random', 'omniscient']:  # ['omniscient', 'random']:
                            if init_method == 'random':
                                pys = [
                                    "main_clustering_diffdim_random.py",
                                    "main_clustering_diffrad_random.py",
                                    "main_clustering_diffvar_random.py",
                                    "main_clustering_diffprop_random.py",
                                ]
                            else:
                                pys = [
                                    "main_clustering_diffdim.py",
                                    "main_clustering_diffrad.py",
                                    "main_clustering_diffvar.py",
                                    "main_clustering_diffprop.py",
                                ]
                            for py in pys:
                                cnt += 1
                                _std = str(std).replace('.', '')
                                _out_dir = f"{OUT_DIR}/std_{_std}/R_{n_repeats}-S_{true_cluster_size}-O_{with_outlier}-B_{n_neighbours}/{init_method}/{py}".replace(
                                    '.', '_')

                                cmd = f"python3 {py} --n_repeats {n_repeats} --true_cluster_size {true_cluster_size} " \
                                      f"--with_outlier {with_outlier} --init_method {init_method} --out_dir {_out_dir} " \
                                      f"--std {std} --n_neighbours {n_neighbours}"

                                log_file = f"{_out_dir}/log.txt"

                                # check if the given directory exists; otherwise, create
                                check_dir(os.path.dirname(log_file))

                                # print(f"{cnt}: {cmd} > {log_file} &")
                                name = f"R_{n_repeats}-S_{true_cluster_size}-Init_{init_method}-{py}"
                                generate_sh(name, cmd, log_file)

    print(f'\n***total submitted jobs for synthetic datasets: {cnt}')

    # for real datasets
    for data_name in ['letter_recognition',
                      'pen_digits']:  # 'music_genre', 'iot_intrusion','iot_intrusion','pen_digits', 'biocoin_heist','letter_recognition']:
        for fake_label in ['random', 'special']:  # ['synthetic', 'random', 'special']:   # False
            out_dir = os.path.join(OUT_DIR, data_name, f'F_{fake_label}')
            for n_repeats in [args.n_repeats]:  # [5000]
                for true_cluster_size in [100]:
                    for std in [0]:  # [0.1, 0.25, 0.5, 1, 0.1, 0.25, ]:
                        for n_neighbours in [0]:
                            for with_outlier in [True]:  # [True, False]:
                                for init_method in ['random', 'omniscient']:  # ['omniscient', 'random']:
                                    if init_method == 'random':
                                        pys = [
                                            "main_clustering_diffprop_random_real.py",
                                            # "main_clustering_diffprop_random_real2.py",
                                        ]
                                    else:
                                        pys = [
                                            "main_clustering_diffprop_real.py",
                                            # "main_clustering_diffprop_real2.py",
                                        ]
                                    for py in pys:
                                        cnt += 1
                                        _std = str(std).replace('.', '')
                                        _out_dir = f"{out_dir}/std_{_std}/R_{n_repeats}-S_{true_cluster_size}-O_{with_outlier}-B_{n_neighbours}/{init_method}/{py}".replace(
                                            '.', '_')

                                        cmd = f"python3 {py} --n_repeats {n_repeats} --true_cluster_size {true_cluster_size} " \
                                              f"--with_outlier {with_outlier} --init_method {init_method} --out_dir {_out_dir} " \
                                              f"--std {std} --data_name {data_name} --fake_label {fake_label} --n_neighbours {n_neighbours}"

                                        log_file = f"{_out_dir}/log.txt"
                                        # check if the given directory exists; otherwise, create
                                        check_dir(os.path.dirname(log_file))

                                        # print(f"{cnt}: {cmd} > {log_file} &")
                                        name = f"R_{n_repeats}-S_{true_cluster_size}-Init_{init_method}-{py}"
                                        generate_sh(name, cmd, log_file)
    print(f'\n***total submitted jobs for real datasets: {cnt}')

    print(f'\n***{cnt} commands in total.')

    return


if __name__ == '__main__':
    main()

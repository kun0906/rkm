"""
Run experiment on HPC

    ssh ky8517@tiger.princeton.edu

    cd /scratch/gpfs/ky8517/rkm/src
    module purge
    module load anaconda3/2021.11
    #conda env list
    #conda create --name py3104_rkm python=3.10.4
    conda activate py3104
    #pip install -r ../requirements.txt  # you should install in the login node (not in compute nodes)

    pwd
    python3 -V
    uname -a

    python3 hpc_sbatch.py


    # download the remote results to local
    rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/rkm/src/out .

    squeue --format="%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R" --me
"""

import argparse
import itertools
import os
import os.path
import subprocess
from functools import partial

import numpy as np

print = partial(print, flush=True)

parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--n_repetitions", type=int, default=5000)  #
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
cd /scratch/gpfs/ky8517/rkm/src_best
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
#conda activate py3104
#pip install -r ../requirements.txt 

pwd
python3 -V
uname -a 


{cmd} &> {log_file} 

# python3 main_diff_dim_random.py --n_repetitions n_repetitions --true_single_cluster_size true_single_cluster_size --init_method init_method &> 'out/main_clustering_diffdim_{name}.txt' & 
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
    for n_repetitions in [args.n_repetitions]:  # [5000]
        for true_single_cluster_size in [100]:
            # for std in [2]:  # [0.5, 1, 2]: #[0.1, 0.25, 0.5, 1, 0.1, 0.25, ]:
            for std in [5]: #[2, 5, 10, 20]:  # [0.5, 1, 2]: #[0.1, 0.25, 0.5, 1, 0.1, 0.25, ]:
                for add_outlier in [True]:  # [True, False]:
                    n_neighbors, theta, m = 0, 0, 0
                    for init_method in ['omniscient', 'random', 'robust_init']:  # ['omniscient', 'random']:
                        pys = [
                            "main_diff_dim.py",
                            "main_diff_rad.py",
                            "main_diff_var.py",
                            "main_diff_prop.py",
                        ]
                        for py in pys:
                            cnt += 1
                            _std = str(std).replace('.', '')
                            _out_dir = f"{OUT_DIR}/cluster_std_2_radius_{_std}/R_{n_repetitions}-S_{true_single_cluster_size}-O_{add_outlier}-B_{n_neighbors}-t_{theta}-m_{m}/{init_method}/{py}".replace(
                                '.', '_')

                            cmd = f"python3 {py} --n_repetitions {n_repetitions} --true_single_cluster_size {true_single_cluster_size} " \
                                  f"--add_outlier {add_outlier} --init_method {init_method} --out_dir {_out_dir} " \
                                  f"--cluster_std 2 --radius {std} --n_neighbors {n_neighbors} --theta {theta} --m {m}"

                            log_file = f"{_out_dir}/log.txt"

                            # check if the given directory exists; otherwise, create
                            check_dir(os.path.dirname(log_file))
                            # os.makedirs(os.path.dirname(log_file), exist_ok=True)

                            # print(f"{cnt}: {cmd} > {log_file} &")
                            # name = f"R_{n_repetitions}-S_{true_single_cluster_size}-Init_{init_method}-{py}"
                            name = py.split('_')[2]
                            name = f"{name}-{init_method[:3]}-R{n_repetitions}"
                            generate_sh(name, cmd, log_file)

    print(f'\n***total submitted jobs for synthetic datasets: {cnt}')

    # # for real datasets
    # for data_name in ['letter_recognition',
    #                   'pen_digits']:  # 'music_genre', 'iot_intrusion','iot_intrusion','pen_digits', 'biocoin_heist','letter_recognition']:
    #     for fake_label in ['random', 'special']:  # ['synthetic', 'random', 'special']:   # False
    #         out_dir = os.path.join(OUT_DIR, data_name, f'F_{fake_label}')
    #         for n_repetitions in [args.n_repetitions]:  # [5000]
    #             for true_single_cluster_size in [100]:
    #                 for std in [0]:  # [0.1, 0.25, 0.5, 1, 0.1, 0.25, ]:
    #                     for n_neighbors in [15]:
    #                         for add_outlier in [True]:  # [True, False]:
    #                             for init_method in ['random', 'omniscient']:  # ['omniscient', 'random']:
    #                                 pys = [
    #                                     "main_diff_prop_real.py",
    #                                     # "main_diffprop_real2.py",
    #                                 ]
    #                                 for py in pys:
    #                                     cnt += 1
    #                                     _std = str(std).replace('.', '')
    #                                     _out_dir = f"{out_dir}/cluster_std_{_std}/R_{n_repetitions}-S_{true_single_cluster_size}-O_{add_outlier}-B_{n_neighbors}/{init_method}/{py}".replace(
    #                                         '.', '_')
    #
    #                                     cmd = f"python3 {py} --n_repetitions {n_repetitions} --true_single_cluster_size {true_single_cluster_size} " \
    #                                           f"--add_outlier {add_outlier} --init_method {init_method} --out_dir {_out_dir} " \
    #                                           f"--cluster_std {std} --data_name {data_name} --fake_label {fake_label} --n_neighbors {n_neighbors}"
    #
    #                                     log_file = f"{_out_dir}/log.txt"
    #                                     # check if the given directory exists; otherwise, create
    #                                     check_dir(os.path.dirname(log_file))
    #
    #                                     # print(f"{cnt}: {cmd} > {log_file} &")
    #                                     # name = f"R_{n_repetitions}-S_{true_single_cluster_size}-Init_{init_method}-{py}"
    #                                     name = py.split('_')[2]
    #                                     name = f"{name}-{init_method[:3]}-R{n_repetitions}"
    #                                     generate_sh(name, cmd, log_file)
    # print(f'\n***total submitted jobs for real datasets: {cnt}')

    print(f'\n***{cnt} commands in total.')

    return


if __name__ == '__main__':
    main()

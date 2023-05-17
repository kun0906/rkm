import os.path
import shutil
import subprocess


def generate_sh():
    out_dir = 'sh'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    radiuses = [1, 2, 3, 4, 5, 6]
    noise_covs = [1, 4, 9, 16, 25]  # [25]
    noise_means = [1, 4, 9, 16, 25]
    props = [0.01, 0.05, 0.1, 0.15, 0.2]  # [0.1, 0.5, 1, 2, 3]

    cnt = 0
    for initial_method in ['omniscient', 'random']:
        for radius in radiuses:  #:
            for noise_mean in noise_means:
                for noise_cov in noise_covs:
                    for prop in props:
                        cnt += 1
                        name = f"r_{radius}-nm_{noise_mean}-nc_{noise_cov}-p_{prop}-{initial_method}"
                        hh, mm = divmod(cnt, 60)
                        s = fr"""#!/bin/bash

#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=24:{mm}:00          # total run time limit (HH:MM:SS)
#SBATCH --output={out_dir}/out_{name}.txt
#SBATCH --error={out_dir}/err_{name}.txt

module purge
cd /scratch/gpfs/ky8517/rkm/rkm
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
# conda activate py3104

pwd
python3 -V
uname -a 

PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_clustering_hpc.py --radius {radius} --nmean {noise_mean} --ncov {noise_cov} --prop {prop} --init_method {initial_method}> '{out_dir}/out_{name}.txt' 2>&1
# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    """
                        out_sh = f'{out_dir}/{name}.sh'
                        # print(out_sh)
                        with open(out_sh, 'w') as f:
                            f.write(s)

                        cmd = f'sbatch {out_sh}'
                        ret = subprocess.run(cmd, shell=True)
                        print(cmd, ret)

    return cnt


if __name__ == '__main__':
    cnt = generate_sh()
    print(f'\n***total submitted jobs: {cnt}')

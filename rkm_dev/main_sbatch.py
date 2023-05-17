import os.path
import shutil
import subprocess


def generate_sh():
    out_dir = 'sh'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    for n_repeats in [500, 1000, 5000]:
        for true_cluster_size in [200]:
            for init_method in ['omniscient', 'random']:
                cnt += 1
                name = f"R_{n_repeats}-S_{true_cluster_size}-Init_{init_method}"
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

python3 main_clustering_diffdim.py --n_repeats {n_repeats} --true_cluster_size {true_cluster_size} --init_method {init_method} &> 'out/main_clustering_diffdim_{name}.txt' & 
python3 main_clustering_diffrad.py --n_repeats {n_repeats} --true_cluster_size {true_cluster_size} --init_method {init_method} &> 'out/main_clustering_diffrad_{name}.txt' & 
python3 main_clustering_diffvar.py --n_repeats {n_repeats} --true_cluster_size {true_cluster_size} --init_method {init_method} &> 'out/main_clustering_diffvar_{name}.txt' & 


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

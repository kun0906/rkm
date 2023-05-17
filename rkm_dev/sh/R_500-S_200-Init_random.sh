#!/bin/bash

#SBATCH --job-name=R_500-S_200-Init_random         # create a short name for your job
#SBATCH --time=24:2:00          # total run time limit (HH:MM:SS)
#SBATCH --output=sh/out_R_500-S_200-Init_random.txt
#SBATCH --error=sh/err_R_500-S_200-Init_random.txt

module purge
cd /scratch/gpfs/ky8517/rkm/rkm
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
# conda activate py3104

pwd
python3 -V
uname -a 

python3 main_clustering_diffdim.py --n_repeats 500 --true_cluster_size 200 --init_method random &> 'out/main_clustering_diffdim_R_500-S_200-Init_random.txt' & 
python3 main_clustering_diffrad.py --n_repeats 500 --true_cluster_size 200 --init_method random &> 'out/main_clustering_diffrad_R_500-S_200-Init_random.txt' & 
python3 main_clustering_diffvar.py --n_repeats 500 --true_cluster_size 200 --init_method random &> 'out/main_clustering_diffvar_R_500-S_200-Init_random.txt' & 


# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    
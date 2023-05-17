#!/bin/bash

cd /u/ky8517/rkm/rkm_final
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
# conda activate py3104

n_repeats=1000
true_cluster_size=100

out_dir="out-R_${n_repeats}-S_${true_cluster_size}"
if [ ! -d $out_dir ]; then
  mkdir $out_dir
fi

pwd
python3 -V
uname -a 

# omniscient
python3 main_clustering_diffdim.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method omniscient --out_dir $out_dir &> "${out_dir}/main_clustering_diffdim_R_${n_repeats}-S_${true_cluster_size}-Init_omniscient.txt" &
python3 main_clustering_diffrad.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method omniscient --out_dir $out_dir &> "${out_dir}/main_clustering_diffrad_R_${n_repeats}-S_${true_cluster_size}-Init_omniscient.txt" &
python3 main_clustering_diffvar.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method omniscient --out_dir $out_dir &> "${out_dir}/main_clustering_diffvar_R_${n_repeats}-S_${true_cluster_size}-Init_omniscient.txt" &

# init_method=random
python3 main_clustering_diffdim_random.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method random --out_dir $out_dir &> "${out_dir}/main_clustering_diffdim_R_${n_repeats}-S_${true_cluster_size}-Init_random.txt" &
python3 main_clustering_diffrad_random.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method random --out_dir $out_dir &> "${out_dir}/main_clustering_diffrad_R_${n_repeats}-S_${true_cluster_size}-Init_random.txt" &
python3 main_clustering_diffvar_random.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method random --out_dir $out_dir &> "${out_dir}/main_clustering_diffvar_R_${n_repeats}-S_${true_cluster_size}-Init_random.txt" &


# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    
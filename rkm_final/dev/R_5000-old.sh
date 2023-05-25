#!/bin/bash

cd /u/ky8517/rkm/rkm_final
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
# conda activate py3104

n_repeats=5000
true_cluster_size=100
with_outlier=True
name="R_${n_repeats}-S_${true_cluster_size}-O_${with_outlier}"
out_dir="out_${name}"
if [ ! -d $out_dir ]; then
  mkdir $out_dir
fi

pwd
python3 -V
uname -a 

#pys=("main_clustering_diffdim.py" "main_clustering_diffrad.py" "main_clustering_diffvar.py")
pys=("main_clustering_diffprop.py")
redirect="&>"
init_method="omniscient"
# The @ symbol in the square brackets indicates that you are looping through all of the elements in the array.
for py in "${pys[@]}"; do
  # https://stackoverflow.com/questions/45499606/how-to-redirect-python-script-cmd-output-to-a-file
  #    cmd="(python3 $py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method $init_method --with_outlier $with_outlier --out_dir $out_dir) &> \"${out_dir}/main_clustering_diffdim_${name}-Init_omniscient.txt\" "
  cmd="python3 $py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method $init_method --with_outlier $with_outlier --out_dir $out_dir"
  echo $cmd
  $cmd &> "${out_dir}/main_clustering_diffdim_${name}-Init_${init_method}.txt" &
done


#pys=("main_clustering_diffdim_random.py" "main_clustering_diffrad_random.py" "main_clustering_diffvar_random.py")
pys=("main_clustering_diffprop_random.py")
init_method="random"
# The @ symbol in the square brackets indicates that you are looping through all of the elements in the array.
for py in "${pys[@]}"; do
  # https://stackoverflow.com/questions/45499606/how-to-redirect-python-script-cmd-output-to-a-file
  cmd="python3 $py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method $init_method --with_outlier $with_outlier --out_dir $out_dir"
  echo $cmd
  $cmd &> "${out_dir}/main_clustering_diffdim_${name}-Init_${init_method}.txt" &
done


## omniscient
#python3 main_clustering_diffdim.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method omniscient --with_outlier $with_outlier --out_dir $out_dir &> "${out_dir}/main_clustering_diffdim_${name}-Init_omniscient.txt" &
#python3 main_clustering_diffrad.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method omniscient --with_outlier $with_outlier --out_dir $out_dir &> "${out_dir}/main_clustering_diffrad_${name}-Init_omniscient.txt" &
#python3 main_clustering_diffvar.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method omniscient --with_outlier $with_outlier --out_dir $out_dir &> "${out_dir}/main_clustering_diffvar_${name}-Init_omniscient.txt" &
#
## init_method=random
#python3 main_clustering_diffdim_random.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method random --with_outlier $with_outlier --out_dir $out_dir &> "${out_dir}/main_clustering_diffdim_${name}-Init_random.txt" &
#python3 main_clustering_diffrad_random.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method random --with_outlier $with_outlier --out_dir $out_dir &> "${out_dir}/main_clustering_diffrad_${name}-Init_random.txt" &
#python3 main_clustering_diffvar_random.py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method random --with_outlier $with_outlier --out_dir $out_dir &> "${out_dir}/main_clustering_diffvar_${name}-Init_random.txt" &


# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    
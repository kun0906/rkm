#!/bin/bash

#cd /u/ky8517/rkm/rkm_final
#module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
#conda activate py3104

pwd
python3 -V
uname -a

date

dims=(2 10)
stds=(0.1 1)
n_repeats=5000
redirect="&>"
# The @ symbol in the square brackets indicates that you are looping through all of the elements in the array.
for dim in "${dims[@]}"; do
  for std in "${stds[@]}"; do
    # https://stackoverflow.com/questions/45499606/how-to-redirect-python-script-cmd-output-to-a-file
    #    cmd="(python3 $py --n_repeats $n_repeats --true_cluster_size $true_cluster_size --init_method $init_method --with_outlier $with_outlier --out_dir $out_dir) &> \"${out_dir}/main_clustering_diffdim_${name}-Init_omniscient.txt\" "
    cmd="python3 main_clustering_diffrad_random_test.py --n_repeats $n_repeats --std $std --dim $dim"
    echo $cmd
    $cmd &> "out/$D_${dim}-rp_${n_repeats}-std_${std}.txt" &
  done
done


#
#cmd="python3 -u main_clustering_diffrad_random_test.py"
#echo $cmd
#$cmd &> log.txt

date

wait
echo 'done'     
    
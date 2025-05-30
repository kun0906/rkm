#!/bin/bash

conda activate py3104_rkm

OUT_DIR="out"
cnt=0
n_repetitions=50
true_single_cluster_size=100
add_outlier=true

n_neighbors=0
theta=0
m=0

for data_name in "letter_recognition" "pen_digits"; do
  for init_method in "omniscient" "random" "robust_init"; do
    py="main_diff_prop_real.py"
    for fake_label in "OMC" "OOC"; do
      ((cnt++))
      # Replace periods in file name with underscores for compatibility
      py_cleaned=${py//./_}
      _out_dir="${OUT_DIR}/R_${n_repetitions}-S_${true_single_cluster_size}-O_${add_outlier}-${fake_label}-B_${n_neighbors}-t_${theta}-m_${m}/${data_name}/${init_method}/${py_cleaned}"

      cmd="python3 ${py} \
        --n_repetitions ${n_repetitions} \
        --true_single_cluster_size ${true_single_cluster_size} \
        --add_outlier ${add_outlier} \
        --init_method ${init_method} \
        --out_dir ${_out_dir} \
        --cluster_std 2 \
        --data_name ${data_name} \
        --n_neighbors ${n_neighbors} \
        --theta ${theta} \
        --m ${m} \
        --fake_label ${fake_label}"   # Outliers from Multiple Classes (OMC) or Outliers from One Class (OOC)

      log_file="${_out_dir}/log.txt"

      mkdir -p "${_out_dir}"
      echo "Running: $cmd"
      $cmd > "${log_file}" 2>&1 &
    done
  done
done

wait  # Wait for all background jobs to complete
echo "All jobs finished."

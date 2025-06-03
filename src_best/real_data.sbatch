#!/bin/bash
#SBATCH --job-name=real_data_array
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --array=0-11
#SBATCH --time=40:30:00
#SBATCH --output=out/logs/out_%A_%a.txt
#SBATCH --error=out/logs/err_%A_%a.txt
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Move to project directory and activate environment
cd ~/rkm/src_best
source activate py3104_rkm

# Show context
date
hostname
which python
python3 -V
uname -a

# Parameters (shared across jobs)
OUT_DIR="out"
n_repetitions=201
true_single_cluster_size=100
add_outlier=True
n_neighbors=0
theta=0
m=0
py="main_diff_prop_real.py"
py_cleaned=${py//./_}

# Define task list inline (space-separated strings)
TASKS=(
  "letter_recognition omniscient OMC"
  "letter_recognition omniscient OOC"
  "letter_recognition random OMC"
  "letter_recognition random OOC"
  "letter_recognition robust_init OMC"
  "letter_recognition robust_init OOC"
  "pen_digits omniscient OMC"
  "pen_digits omniscient OOC"
  "pen_digits random OMC"
  "pen_digits random OOC"
  "pen_digits robust_init OMC"
  "pen_digits robust_init OOC"
)
#
#
## Define task list inline (space-separated strings)
#TASKS=(
#  "letter_recognition omniscient OMC"
#)

# Extract current task
IFS=' ' read -r data_name init_method fake_label <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

# Build output directory
_out_dir="${OUT_DIR}/R_${n_repetitions}-S_${true_single_cluster_size}-O_${add_outlier}-${fake_label}-B_${n_neighbors}-t_${theta}-m_${m}/${data_name}/${init_method}/${py_cleaned}"
mkdir -p "${_out_dir}"

# Construct and run the command
cmd="python3 ${py} \
  --n_repetitions ${n_repetitions} \
  --true_single_cluster_size ${true_single_cluster_size} \
  --add_outlier ${add_outlier} \
  --init_method ${init_method} \
  --out_dir ${_out_dir} \
  --cluster_std 0 \
  --data_name ${data_name} \
  --n_neighbors ${n_neighbors} \
  --theta ${theta} \
  --m ${m} \
  --fake_label ${fake_label}"

echo "Running: $cmd"
$cmd > "${_out_dir}/log.txt" 2>&1

echo "Finished task $SLURM_ARRAY_TASK_ID"

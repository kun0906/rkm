#!/bin/bash
#SBATCH --job-name=real_data_array
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --array=0-71
#SBATCH --time=48:00:00
#SBATCH --output=out/logs/out_%A_%a.txt
#SBATCH --error=out/logs/err_%A_%a.txt
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Move to project directory. If that fails (e.g., directory doesn't exist),
# immediately exit the script with status code 1 (indicating an error).
#cd ~/rkm/src _robust || { echo "❌ Failed to cd into project directory"; exit 1; }
pwd
module load conda
which conda
conda -V
conda env list
conda activate py3104_rkm || { echo "Failed to activate conda environment py3104_rkm"; exit 1; }
echo

# Show context
date
hostname
which python
python3 -V
uname -a

# Parameters (shared across jobs)
OUT_DIR="out"
n_repetitions=100
true_single_cluster_size=100
add_outlier=True
n_neighbors=0
theta=0
m=0
py="main_diff_prop_real.py"
py_cleaned=${py//./_}

# Initialize an empty array
TASKS=()
cnt=0
for dataset in letter_recognition pen_digits; do
  for fake_label in OMC OOC; do
    for clustering_method in k_medians_l2 k_medians_l1 k_means \
        robust_lp_k_medians_l2 robust_lp_k_medians_l1 robust_lp_k_means; do
      for init_method in random omniscient; do    # robust_init
        TASKS+=("$dataset $fake_label $clustering_method $init_method")
        ((cnt++))  # increment counter correctly
      done
    done
  done
done
echo "Total number of tasks: $cnt"

## Define task list inline (space-separated strings)
echo "TASKS:"
for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  echo "$idx: $task"
done

echo
# Read into individual variables
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "TASKS[SLURM_ARRAY_TASK_ID]: ${TASKS[$SLURM_ARRAY_TASK_ID]}"
IFS=' ' read -r data_name fake_label clustering_method init_method <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
# Optional: print values for debugging
echo "data_name: $data_name"
echo "fake_label: $fake_label"
echo "clustering_method: $clustering_method"
echo "init_method: $init_method"

# Build output directory path in multiple lines for readability
_out_dir="${OUT_DIR}/R_${n_repetitions}-S_${true_single_cluster_size}-O_${add_outlier}-B_${n_neighbors}-t_${theta}-m_${m}/\
${data_name}/\
${fake_label}/\
${clustering_method}/\
${init_method}/\
${py_cleaned}"

# Create the output directory if it doesn't exist
mkdir -p "${_out_dir}"

# Construct and run the command
cmd="python3 ${py} \
  --clustering_method ${clustering_method} \
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

echo
echo "✅ Finished task $SLURM_ARRAY_TASK_ID"
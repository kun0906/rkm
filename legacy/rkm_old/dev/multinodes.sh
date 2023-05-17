#!/bin/bash
#SBATCH --job-name=cxx_mpi_omp   # create a short name for your job
#SBATCH --output=out_%x_%j_%N_%n.txt
#SBATCH --error=err_%x_%j_%N_%n.txt
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=3      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
## SBATCH --mail-type=end          # send email when job ends
## SBATCH --mail-type=fail         # send mail if job fails
## SBATCH --mail-user=ky8517@princeton.edu

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

date

module purge
cd /scratch/gpfs/ky8517/rkm/rkm/dev
module load anaconda3/2021.11

python3 -V
### all print info in python will be outputted to #SBATCH --output=out_%x_%j_%N_%n_%t.txt
#srun python3 task1.py
#srun python3 task2.py

## all print info in python will be outputted to #SBATCH --output=out_%x_%j_%N_%n_%t.txt
#srun python3 task1.py &
#srun python3 task2.py &

# https://unix.stackexchange.com/questions/170572/what-is-in-a-shell-script#:~:text=%26%20means%20both%20standard%20output%20(%201,Eg.
# program &>> result.txt is equivalent to program >> result.txt 2>&1
srun python3 task1.py &> task1.txt &
srun python3 task2.py &> task2.txt &


## combine > and & to the end of srun, not working
#srun python3 task1.py > task1.txt 2>&1 &
#srun python3 task2.py > task2.txt 2>&1 &

# only run on one node
#python3 task1.py > task1.txt 2>&1
#python3 task2.py > task2.txt 2>&1

wait

date
echo 'done'
#!/bin/bash

#SBATCH --job-name=rkm         # create a short name for your job
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=out/sh/out.txt
#SBATCH --error=out/sh/err.txt
### SBATCH --mail-type=end          # send email when job ends
###SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification.
## SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
## SBATCH --output=out/sh/%j-2gaussians-300-(3,7)-1000-out.txt
## SBATCH --error=out/sh/%j-2gaussians-300-(3,7)-1000-err.txt
## SBATCH --nodes=1                # node count
## SBATCH --ntasks=1               # total number of tasks across all nodes
## SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
## SBATCH --mem=40G\
### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-user=kun.bj@cloud.com # not work
### SBATCH --mail-user=<YourNetID>@princeton.edu\

module purge
cd /scratch/gpfs/ky8517/rkm/rkm
module load anaconda3/2021.11
#conda create --name py3104 python=3.10.4
#conda activate py3104
#pip3 install -r requirement.txt

pwd
python3 -V

# sshfs ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/rkm rkm
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py > 'out/sh/log.txt' 2>&1
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_clustering.py > 'out/log.txt' 2>&1
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_clustering_hpc.py > 'out/log2.txt' 2>&1
#### sbatch ./sbatch.sh
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py > 'out/sh/log.txt' 2>&1  &
# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

# checkquota

#wait
#echo $!
#echo $?

echo 'done'

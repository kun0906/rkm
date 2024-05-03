#!/bin/bash

cd /u/ky8517/rkm/rkm_final
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
#conda activate py3104

pwd
python3 -V
uname -a

date
cmd="python3 process_batch.py"    # for synthetic datasets
echo $cmd
$cmd &> log_synthetic.txt &

cmd="python3 process_batch_real.py" # for real datasets
echo $cmd
$cmd &> log_real.txt &
date

wait
echo 'done'     


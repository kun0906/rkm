#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux
'
#ssh ky8517@tigergpu.princeton.edu
cd /scratch/gpfs/ky8517/rkm/rkm
module load anaconda3/2021.11

srun --time=20:00:00 --pty bash -i
#srun --nodes=1  --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#cd /scratch/gpfs/ky8517/rkm/rkm
#module load anaconda3/2021.11

#sshfs ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/rkm tiger

#!/bin/bash

module load anaconda3/2021.11
cd /scratch/gpfs/ky8517/rkm/rkm
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py

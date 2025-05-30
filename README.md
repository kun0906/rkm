## Robust K-Means Clustering Algorithm

**

## Table of Contents

* [Environment requirement](#Environment)
* [Installation](#Installation)
* [Usage](#Usage)
* [Project structure](#Project)
* [Dataset](#Dataset)
* [Contact](#contact)

<!-- * [License](#license) -->

## Environment requirement <a name="Environment"></a>

- Conda 4.10.3 # conda -V
  - conda activate py3104_rkm
- Python 3.10.4 # python3 -V
- Pip3 22.1.2 # pip3 -V

## Installation  <a name="Installation"></a>

`$pip3 install -r requirements.txt`

## Project structure <a name="Project"></a>

- docs
- rkm
    - data
    - cluster
    - utils
    - vis
    - out
- requirement.txt
- README.md
- UPDATE.md

## Data:

[//]: # (- GAUSSIAN3: _simulated 2 clusters from 2 Gaussian distributions._)

[//]: # (- MNIST: _handwritten datasets: https://yann.lecun.com/exdb/mnist/ or https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/index.html)

[//]: # (- NBAIOT: _IoT dataset: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)

## Usage

```shell
$ssh ky8517@tiger.princeton.edu
$tmux ls
$tmux new -s rkm
#tmux attach -t rkm
$cd /scratch/gpfs/ky8517/rkm/src
$module purge
$module load anaconda3/2021.11
#conda env list
#conda create --name py3104_rkm python=3.10.4
#conda activate py3104
#pip install -r ../requirements.txt  # you should install in the login node (not in compute nodes)
$python3 hpc_sbatch.py

ssh kunyang@slogin-01.superpod.smu.edu
srun -A kunyang_nvflare_py31012_0001 -G 1 -t 800 --nodelist=bcm-dgxa100-0018 --pty $SHELL
conda activate py3104_rkm
./run_real_data.sh

```

[//]: # ()

[//]: # (Note that in R_5000.sh,)

[//]: # (_process_batch.py &#40;for synthetic datasets&#41; and  )

[//]: # (process_batch_real.py &#40;for real-world datasets&#41;_)

[//]: # ()

[//]: # ($ssh ky8517@nobel.princeton.edu)

[//]: # ($sshfs ky8517@nobel.princeton.edu:/u/ky8517/ nobel -o volname=nobel)

[//]: # ($killall -u ky8517)

[//]: # (# download all from the remote server)

[//]: # ()

[//]: # ($rsync -azP ky8517@nobel.princeton.edu:/u/ky8517/rkm/rkm_final/out .)

## Update

- All the update details can be seen in UPDATE.md

## Contact

- Email: kun88.yang@gmail.com

[//]: #

[//]: #

[//]: #

[//]: #

[//]: #

[//]: #

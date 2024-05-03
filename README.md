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
- Python 3.10.4 # python3 -V
- Pip3 22.1.2 # pip3 -V

## Installation  <a name="Installation"></a>
  `$pip3 install -r requirements.txt`

## Project structure <a name="Project"></a>

- docs
- rkm
  - datasets
  - cluster
  - utils
  - vis
  - out
- requirement.txt
- README.md
- UPDATE.md

## Dataset:

[//]: # (- GAUSSIAN3: _simulated 2 clusters from 2 Gaussian distributions._)

[//]: # (- MNIST: _handwritten datasets: https://yann.lecun.com/exdb/mnist/ or https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/index.html)

[//]: # (- NBAIOT: _IoT dataset: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)

## Usage

```shell
$ssh ky8517@nobel.princeton.edu
$tmux ls
$tmux new -s rkm
#tmux attach -t rkm
$cd rkm/rkm_final/
#$PYTHONPATH='..' python3 main_all.py
./R_5000.sh
```
Note that in R_5000.sh, 
_process_batch.py (for synthetic datasets) and  
process_batch_real.py (for real-world datasets)_ 

$ssh ky8517@nobel.princeton.edu
$sshfs ky8517@nobel.princeton.edu:/u/ky8517/ nobel -o volname=nobel
$killall -u ky8517

# download all from the remote server
$rsync -azP ky8517@nobel.princeton.edu:/u/ky8517/rkm/rkm_final/out .
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

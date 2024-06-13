"""

"""
import argparse
import os
import subprocess
from tqdm import tqdm
from functools import partial

print = partial(print, flush=True)

parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--n_repetitions", type=int, default=2)  #
args = parser.parse_args()
print(args)


# project_dir = '~/'
# os.chdir(project_dir)

def check_dir(in_dir):
    # if os.path.isfile(in_pth):
    #     # To find whether a given path is an existing regular file or not.
    #     in_dir = os.path.dirname(in_pth)
    # elif os.path.isdir(in_pth):
    #     in_dir = in_pth
    # else:
    #     raise ValueError(in_pth)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

    return


n_max_process = 48  # the maximum number of processes that can be running at the same time.


def main():
    # out_dir = "out_letter_recognition"
    # out_dir = "out_BitcoinHeistData"
    cnt = 0
    procs = set()
    OUT_DIR = 'out'
    for data_name in ['letter_recognition',
                      'pen_digits']:  # 'music_genre', 'iot_intrusion','iot_intrusion','pen_digits', 'biocoin_heist','letter_recognition']:
        for fake_label in ['random', 'special']:  # ['synthetic', 'random', 'special']:   # False
            out_dir = os.path.join(OUT_DIR, data_name, f'F_{fake_label}')
            for n_repetitions in [args.n_repetitions]:  # [5000]
                for true_single_cluster_size in [100]:
                    for std in [0]:  # [0.1, 0.25, 0.5, 1, 0.1, 0.25, ]:
                        for add_outlier in [True]:  # [True, False]:
                            for init_method in ['random', 'omniscient']:  # ['omniscient', 'random']:
                                if init_method == 'random':
                                    pys = [
                                        # "main_clustering_diffdim_random.py",
                                        # "main_clustering_diffrad_random.py",
                                        # "main_clustering_diffvar_random.py",
                                        "main_clustering_diffprop_random_real.py",
                                        # "main_clustering_diffprop_random_real2.py",
                                    ]
                                else:
                                    pys = [
                                        # "main_clustering_diffdim.py",
                                        # "main_clustering_diffrad.py",
                                        # "main_clustering_diffvar.py",
                                        "main_clustering_diffprop_real.py",
                                        # "main_clustering_diffprop_real2.py",
                                    ]
                                for py in pys:
                                    cnt += 1
                                    _std = str(std).replace('.', '')
                                    _out_dir = f"{out_dir}/std_{_std}/R_{n_repetitions}-S_{true_single_cluster_size}-O_{add_outlier}/{init_method}/{py}".replace(
                                        '.', '_')

                                    cmd = f"python3 {py} --n_repetitions {n_repetitions} --true_single_cluster_size {true_single_cluster_size} " \
                                          f"--add_outlier {add_outlier} --init_method {init_method} --out_dir {_out_dir} " \
                                          f"--cluster_std {std} --data_name {data_name} --fake_label {fake_label}"

                                    log_file = f"{_out_dir}/log.txt"

                                    # check if the given directory exists; otherwise, create
                                    check_dir(os.path.dirname(log_file))

                                    # while len(procs) >= n_max_process:
                                    #     for p in procs:
                                    #         if p.poll() is None:
                                    #             # the process is still running.
                                    #             # print(p.pid)
                                    #             pass
                                    #         else:
                                    #             print(f"{p.pid} finished and returncode was {p.returncode}")
                                    #             procs.remove(p)

                                    print(f"{cnt}: {cmd} > {log_file} &")
                                    with open(log_file, 'w') as f:
                                        p = subprocess.Popen(cmd, stderr=f, stdout=f, shell=True)
                                    procs.add(p)
                                    print(f"pid:{p.pid} started.")

    print(f'\n***{cnt} commands in total.')

    # for i, p in tqdm(enumerate(procs)):
    #     # ret = p.wait()
    #     # https://stackoverflow.com/questions/44456996/does-popen-communicate-implicitly-call-popen-wait
    #     ret = p.communicate()  # communicate() call wait() implicitly
    #     print(ret, p.returncode, p.pid)


if __name__ == '__main__':
    main()

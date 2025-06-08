import argparse
import time
from functools import wraps
import datetime

def timer(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'{func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'{func.__name__} Took {total_time:.4f} seconds')
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        return result

    return timeit_wrapper


def parse_arguments():
    parser = argparse.ArgumentParser(description="This script performs clustering on given data.")
    parser.add_argument("--n_repetitions", type=int, default=5000,
                        help="Number of experiment repetitions for the clustering method.")
    parser.add_argument("--true_single_cluster_size", type=int, default=100)
    parser.add_argument("--init_method", type=str, default='random', choices=['omniscient', 'random'])
    parser.add_argument("--add_outlier", type=str, default='True', choices=['True', 'False'],
                        help='Whether to add outliers to training set or not.')
    parser.add_argument("--out_dir", type=str, default='out')
    parser.add_argument("--cluster_std", type=float, default=2,
                        help='Standard deviation of each cluster (not outlier standard deviation).')
    parser.add_argument("--radius", type=float, default=5.0,
                        help='location of true centroids.')
    parser.add_argument("--n_neighbors", type=int, default=0,
                        help='Number of neighbors to consider when computing robust spectral clustering (RSC).')
    parser.add_argument("--theta", type=int, default=0,
                        help='Number of edges will be removed when computing robust spectral clustering (RSC).')
    parser.add_argument("--m", type=float, default=0,
                        help='For node i, percentage of neighbor nodes will be ignored '
                             'when computing robust spectral clustering (RSC).')
    return parser.parse_args()

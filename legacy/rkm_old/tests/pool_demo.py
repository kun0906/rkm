import time
from multiprocessing import Pool

from rkm.utils.common import timer


def f2(CASE='', py_names=''):
    print(CASE, py_names)
    for i in range(10000000):
        i = i * 2


def print_range(p):
    # print range
    # print('From {} to {}:'.format(p[0], range[1]))
    print(p.items())
    f2(CASE=p['CASE'], py_names=p['py_names'])


@timer
def run_parallel():
    # list of ranges
    # list_ranges = [[0, 10], [10, 20], [20, 30]]

    cases = ['diff2_outliers']
    py_names = [
        'kmeans',
        'kmedian_l1',
        'kmedian',  # our method
        # 'kmedian_tukey',
    ]

    list_ranges = []
    for CASE in cases:  # , 'mixed_clusters', 'diff_outliers', 'constructed_3gaussians', 'constructed2_3gaussians
        for py_name in py_names:
            list_ranges.append({'CASE': CASE, 'py_names': [py_name]})

    # pool object with number of elements in the list
    pool = Pool(processes=len(list_ranges))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(print_range, list_ranges)


@timer
def run_serial():
    cases = ['diff2_outliers']
    py_names = [
        'kmeans',
        'kmedian_l1',
        'kmedian',  # our method
        # 'kmedian_tukey',
    ]
    for CASE in cases:  # , 'mixed_clusters', 'diff_outliers', 'constructed_3gaussians', 'constructed2_3gaussians
        for py_name in py_names:
            f2(CASE, py_name)


# Driver code
if __name__ == '__main__':
    run_parallel()
    time.sleep(10)   # 5s
    run_serial()

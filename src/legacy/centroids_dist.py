import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import ast
SHOW=False
def plot_hist(data, bins=10, title='', centroid_file=''):
    import matplotlib.pyplot as plt

    counts, edges, bars = plt.hist(data, bins=bins)
    # print(sum(bars), len(data))
    plt.bar_label(bars)
    plt.xlabel('Distances')
    plt.ylabel('Frequency')

    f = f'{centroid_file}-{title}.png'
    step = 50
    plt.title('\n'.join([f[i*step:(i+1)*step] for i in range(len(f)//step+1)]))
    plt.savefig(f, dpi=100)
    if SHOW: plt.show()
    plt.close()


def main(centroid_file):
    # centroid_file
    df = pd.read_csv(centroid_file)

    for col in df.columns:
        # print(f'column: {col}')
        n_repetitions = df.shape[0]
        data = []
        for i in range(n_repetitions):
            tmp = df.loc[i, col].replace('\n', '')[2:-2].split('] [') # transform the string to list
            tmp = np.asarray([[float(v) for v in l.split()] for l in tmp]) # transform the list to numpy array
            ds = cdist(tmp, tmp, metric='euclidean')
            ds = ds[np.triu_indices_from(ds, k=1)]    # k=1 means only exclude the main diagonal items.
            data.extend(ds.tolist())
        # 4! = 3*2*1 = 6
        assert 6*n_repetitions == len(data)
        print(f'column: {col}, n_repetitions: {n_repetitions}, total_distances: {len(data)}')
        plot_hist(data, bins=10, title=col, centroid_file=centroid_file)


if __name__ == '__main__':
    for out_dir in ['out/std_01', 'out/std_025','out/std_05','out/std_1']:
        # out_dir = 'out/std_1'
        out_dir = f'{out_dir}/R_5000-S_100-O_True/random/main_clustering_diffrad_random_py'
        num_centroids = 4
        # i = 1
        rad_out = 100.0
        centroid_file = f'{out_dir}/seeds/K_{num_centroids}-R_{rad_out}.csv'
        main(centroid_file)

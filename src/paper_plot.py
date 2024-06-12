"""
https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python



"""
import traceback

import pandas as pd

from base import *


def plot_results(f, out_dir='', out_name='diffdim_random_mp', xlabel='', ylabel='MP', fontsize=10):
    # Plot the line plot with error bars
    df = pd.read_csv(f)
    # Plot the line plot with error bars
    fig, ax = plt.subplots(figsize=(8, 6))

    X_axis = df['x_axis']
    for clustering_method in CLUSTERING_METHODS:
        ls, color, label = LINESTYLES_COLORS_LABELS[clustering_method]
        y, yerr = df[f'{clustering_method}_mp_mu'], df[f'{clustering_method}_mp_std']
        plt.plot(X_axis, y, ls, label=label, color=color)
        plt.errorbar(X_axis, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(X_axis)
    plt.xlabel(xlabel, fontdict={'fontsize': fontsize})
    plt.ylabel(ylabel, fontdict={'fontsize': fontsize})
    # plt.title(title+'_mp')
    plt.legend()
    plt.tight_layout()

    # save the figure
    # temp = time.time()
    # temp = 0
    plt.savefig(f"{out_dir}/{out_name}_mp.png", dpi=300)
    plt.show(block=False)
    # plt.pause(2)


if __name__ == '__main__':
    R = 1001  # 5000  # number of repeats
    S = 100  # each cluster size
    # in_dir = f'out-20240608-m_normalization=True/std_2/R_{R}-S_100-O_True-B_9' # for different m (percentage): 0.1 to 0.9
    in_dir = f'out-20240609-different_projected_k/std_2/R_{R}-S_100-O_True-B_9'  # for different projected_k    : 1 to 9
    out_dir = f'{in_dir}/paper_plot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for alg_method in ['diff_dim', 'diff_rad', 'diff_var',
                       'diff_prop']:  # ['diffdim', 'diffrad', 'diffvar', 'diffprop']:
        for init_method in ['omniscient', 'random']:
            if init_method == 'random':
                py = f"main_{alg_method}_{init_method}_py"
            else:
                py = f"main_{alg_method}_py"
            # if alg_method=='diffrad':
            #     py = py + '-std_01'
            f = f'{in_dir}/{init_method}/{py}/data_4_clusters.csv'
            # f = 'out/data_4_clusters.csv'
            print(f)
            fontsize = 12
            try:
                if alg_method == 'diff_dim':
                    xlabel = 'Dimension'
                elif alg_method == 'diff_var':
                    xlabel = "Outlier Standard Deviation"
                elif alg_method == 'diff_rad':
                    xlabel = "Outlier Location"
                elif alg_method == 'diff_prop':
                    xlabel = "Outlier Proportion"
                else:
                    raise NotImplementedError()

                plot_result(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}_indiv', xlabel=xlabel,
                            fontsize=fontsize)
                plot_results(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}', xlabel=xlabel,
                             fontsize=fontsize)
            except Exception as e:
                traceback.print_exc()

    print('finished')

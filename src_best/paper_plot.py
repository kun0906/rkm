"""
https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python



"""
import os.path
import traceback

import matplotlib.pyplot as plt
import pandas as pd

from base import *

CLUSTERING_METHODS = ['k_medians_l2', 'k_medians_l1', 'k_means',
                      'sc_k_medians_l2', 'sc_k_medians_l1', 'sc_k_means',
                      'rsc_k_medians_l2', 'rsc_k_medians_l1', 'rsc_k_means',
                      # 'rsc_k_means_orig'  # robust k_means from the original api
                      ]


def plot_results(f, out_dir='', out_name='diffdim_random_mp', xlabel='', ylabel='MP', fontsize=10, show=False):
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
    img_pth = f"{out_dir}/{out_name}_mp.png"
    plt.savefig(img_pth, dpi=300)
    if show: plt.show(block=False)
    # plt.pause(2)
    plt.close()
    return img_pth


def merge_images(imgs, out_dir):
    plt.close()
    from PIL import Image
    imgs_lst = []
    max_w, max_h = 0, 0
    for lst in imgs:
        clustering_method, omniscient, random = lst[0], lst[1], lst[2]
        omniscient = Image.open(omniscient)
        random = Image.open(random)
        w, h = omniscient.size
        max_w, max_h = max(max_w, w), max(max_h, h)
        imgs_lst.append((clustering_method, omniscient, random))

    widths, heights =  max_w*2, max_h * len(imgs)
    merged_image = Image.new('RGB', (widths, heights))
    y_offset = 0
    for clustering_method, omniscient, random in imgs_lst:
        merged_image.paste(omniscient, (0,y_offset))
        merged_image.paste(random, (w,y_offset))
        y_offset += h
        omniscient.close()
        random.close()

    merged_image.save(f'{out_dir}/merged_image.jpg')
    # Display the merged image using matplotlib
    plt.imshow(merged_image)
    plt.axis('off')  # Hide axes for cleaner display
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()


def main(tuning_param):
    R = 100  # 5000  # number of repeats
    S = 100  # each cluster size
    # in_dir = f'out-20240608-m_normalization=True/std_2/R_{R}-S_100-O_True-B_9' # for different m (percentage): 0.1 to 0.9
    # in_dir = f'out_best_results_cluster_std=2/cluster_std_10/R_{R}-S_100-O_True-B_0-t_0-m_0'  # for different projected_k    : 1 to 9

    # cluster_std 2 and  different radius of sphere for normal data    : 2, 5, 10, 20
    in_dir = f'out_default_parameters/cluster_std_{tuning_param}/R_{R}-S_100-O_True-B_0-t_0-m_0'
    #
    # in_dir = f'out_best_results_cluster_std=2/cluster_std_{tuning_param}/R_{R}-S_100-O_True-B_0-t_0-m_0'
    # in_dir = f'out_best_results_cluster_std=10/cluster_std_{tuning_param}/R_{R}-S_100-O_True-B_0-t_0-m_0'

    in_dir = f'out_best_params/cluster_std_2_radius_{tuning_param}/R_{R}-S_100-O_True-B_0-t_0-m_0'
    # in_dir = f'out_default_params/cluster_std_2_radius_{tuning_param}/R_{R}-S_100-O_True-B_0-t_0-m_0'

    out_dir = f'{in_dir}/paper_plot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgs = []
    for alg_method in ['diff_dim', 'diff_rad', 'diff_var',
                       'diff_prop']:  # ['diffdim', 'diffrad', 'diffvar', 'diffprop']:
        img_lst = []
        for init_method in ['omniscient', 'random']:  # 'omniscient',
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

                plot_result(pd.read_csv(f), out_dir=os.path.join(out_dir, f'indiv/{init_method}'),
                            out_name=f'{alg_method}_mp',
                            title=f'{alg_method}_{init_method}_indiv', xlabel=xlabel, show=False)
                img = plot_results(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}', xlabel=xlabel,
                                   fontsize=fontsize, show=False)
            except Exception as e:
                traceback.print_exc()

            img_lst.append(img)

        imgs.append([alg_method] + img_lst)

    merge_images(imgs, out_dir)

    print('finished')


if __name__ == '__main__':
    for tuning_param in [2, 5, 10, 20]:
        main(tuning_param)
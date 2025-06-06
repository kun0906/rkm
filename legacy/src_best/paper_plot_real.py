"""
https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python

"""
import os.path
import shutil
import traceback

from paper_plot import plot_results

if __name__ == '__main__':

    # # real data only has diffprop
    # for data_name in ['letter_recognition', 'pen_digits']:  # 'letter_recognition', 'pen_digits'
    #     for fake_label in ['OMC', 'OOC']:  # ['synthetic', 'random', 'special']:
    #
    #         R = 202  # 5000  # number of repeats
    #         S = 100
    #         # in_dir = 'paper_results-20230614/out-outlier_prop_0.6-std_10-normal_std_2/std_2/R_5000-S_100-O_True'
    #         # in_dir = f'paper_results-20230614/real_data_20230803/{data_name}/F_{fake_label}/std_0/R_5000-S_100-O_True'
    #         # in_dir = f'paper_results-20240504/{data_name}/F_{fake_label}/std_0/R_{R}-S_100-O_True'
    #         # in_dir = f'out_SC_beta=0.3_20240508/{data_name}/F_{fake_label}/std_0/R_{R}-S_100-O_True'
    #         in_dir = f'out/R_{R}-S_{S}-O_True-{fake_label}-B_0-t_0-m_0/{data_name}'
    #         out_dir = f'{in_dir}/paper_plot'
    #         if os.path.exists(out_dir):
    #             shutil.rmtree(out_dir)
    #         if not os.path.exists(out_dir):
    #             os.makedirs(out_dir)
    #
    #         for alg_method in ['diff_prop']:  # ['diffdim', 'diffrad', 'diffvar', 'diffprop']:
    #             for init_method in ['omniscient', 'random', 'robust_init']:
    #                 py = f"main_{alg_method}_real_py"
    #                 # if alg_method=='diffrad':
    #                 #     py = py + '-std_01'
    #                 f = f'{in_dir}/{init_method}/{py}/data_3_clusters.csv'
    #                 print(f)
    #                 fontsize = 12
    #                 try:
    #                     if alg_method == 'diff_prop':
    #                         name = 'letter' if data_name == 'letter_recognition' else 'pen'
    #                         xlabel = "Outlier Proportion"
    #                         plot_results(f, out_dir=out_dir, out_name=f'{name}_{fake_label}_{alg_method}_{init_method}',
    #                                      xlabel=xlabel, fontsize=fontsize)
    #
    #                     else:
    #                         raise NotImplementedError()
    #                 except Exception as e:
    #                     traceback.print_exc()
    #
    # print('finished')

    from base import *
    import pandas as pd

    # real data only has diffprop
    for data_name in ['letter_recognition', 'pen_digits']:  # 'letter_recognition', 'pen_digits'
        for fake_label in ['OMC', 'OOC']:  # ['synthetic', 'random', 'special']:

            R = 202  # 5000  # number of repeats
            S = 100
            # in_dir = 'paper_results-20230614/out-outlier_prop_0.6-std_10-normal_std_2/std_2/R_5000-S_100-O_True'
            # in_dir = f'paper_results-20230614/real_data_20230803/{data_name}/F_{fake_label}/std_0/R_5000-S_100-O_True'
            # in_dir = f'paper_results-20240504/{data_name}/F_{fake_label}/std_0/R_{R}-S_100-O_True'
            # in_dir = f'out_SC_beta=0.3_20240508/{data_name}/F_{fake_label}/std_0/R_{R}-S_100-O_True'
            in_dir = f'out/R_{R}-S_{S}-O_True-{fake_label}-B_0-t_0-m_0/{data_name}'
            out_dir = f'{in_dir}/paper_plot'
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            ylabel = 'MP'
            for alg_method in ['diff_prop']:  # ['diffdim', 'diffrad', 'diffvar', 'diffprop']:

                # Plot the line plot with error bars
                fig, ax = plt.subplots(figsize=(8, 6))
                py = f"main_{alg_method}_real_py"
                # if alg_method=='diffrad':
                #     py = py + '-std_01'

                name = 'letter' if data_name == 'letter_recognition' else 'pen'
                xlabel = "Outlier Proportion"

                for init_method_0 in ['random', 'omniscient']:
                    for init_method, clustering_method in [('robust_init', 'k_medians_l2'), (init_method_0, 'k_medians_l1'), (init_method_0, 'k_means')]:
                    # for init_method, clustering_method in [('robust_init', 'robust_lp_k_medians_l2'), (init_method_0, 'robust_lp_k_medians_l1'), (init_method_0, 'robust_lp_k_means')]:
                        f = f'{in_dir}/{init_method}/{py}/data_3_clusters.csv'
                        print(f)
                        fontsize = 12

                        df = pd.read_csv(f)
                        X_axis = df['x_axis']

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
                    out_name = f'{name}_{fake_label}_{init_method_0}'
                    img_pth = f"{out_dir}/{out_name}_mp.png"
                    plt.savefig(img_pth, dpi=300)
                    # if show: plt.show(block=False)
                    # plt.pause(2)
                    plt.close()

    print('finished')

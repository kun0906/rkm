"""
https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python

"""
import os.path
import sys
import time
import traceback

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


def plot_diffdim(f, out_dir='', out_name='diffdim_random_mp', fontsize=10):
    # Plot the line plot with error bars

    # data = {'dimensions': tot_dims, 'lloydL1ians misc': lloydL1_misc_avg, 'lloydL1ians misc err_bar': lloydL1_misc_err,
    #         'lloydL1ians-L1 misc': kmed_misc_avg, 'lloydL1ians-L1 misc err_bar': kmed_misc_err,
    #         'kmeans misc': kmeans_misc_avg, 'kmeans misc err_bar': kmeans_misc_err,
    #         'lloydL1ians acd': lloydL1_acd_avg, 'lloydL1ians acd err_bar': lloydL1_acd_err,
    #         'lloydL1ians-L1 acd': kmed_acd_avg, 'lloydL1ians-L1 acd err_bar': kmed_acd_err,
    #         'kmeans acd': kmeans_acd_avg, 'kmeans acd err_bar': kmeans_acd_err
    #         }

    df = pd.read_csv(f)
    num_centroids = 4
    tot_dims = np.linspace(2, 20, 10).astype(int)

    lloydL1_misc_avg, lloydL1_misc_err = df['lloydL1ians misc'], df['lloydL1ians misc err_bar']
    kmed_misc_avg, kmed_misc_err = df['lloydL1ians-L1 misc'], df['lloydL1ians-L1 misc err_bar']
    kmeans_misc_avg, kmeans_misc_err = df['kmeans misc'], df['kmeans misc err_bar']

    lloydL1_acd_avg, lloydL1_acd_err = df['lloydL1ians acd'], df['lloydL1ians acd err_bar']
    kmed_acd_avg, kmed_acd_err = df['lloydL1ians-L1 acd'], df['lloydL1ians-L1 acd err_bar']
    kmeans_acd_avg, kmeans_acd_err = df['kmeans acd'], df['kmeans acd err_bar']

    figsize = (8, 6)
    fig, ax = plt.subplots()

    plt.plot(tot_dims, lloydL1_misc_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(tot_dims, lloydL1_misc_avg, yerr=lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_dims, kmed_misc_avg, '--', label='k-median', color="purple")
    plt.errorbar(tot_dims, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_dims, kmeans_misc_avg, '-', label='Llyod (k-means)', color="blue")
    plt.errorbar(tot_dims, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(tot_dims)

    plt.xlabel("Dimension",fontdict={'fontsize':fontsize})
    plt.ylabel("MP",fontdict={'fontsize':fontsize})
    # plt.title(title+'_mp')
    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()

    # save the figure
    temp = time.time()
    temp = 0
    plt.savefig(f"{out_dir}/{out_name}_mp.png", dpi=300)

    plt.show(block=False)
    # plt.pause(2)

    # Plot the line plot with error bars
    # figsize = (8, 6)
    fig, ax = plt.subplots()

    plt.plot(tot_dims, lloydL1_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(tot_dims, lloydL1_acd_avg, yerr=lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_dims, kmed_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(tot_dims, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_dims, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(tot_dims, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,lloydL1_acd_avg,kmed_acd_avg))+0.3)
    ax.set_xticks(tot_dims)

    plt.xlabel("Dimension", fontdict={'fontsize':fontsize})
    plt.ylabel("ACD", fontdict={'fontsize':fontsize})
    # plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_sigma_%g" % (num_centroids, radius, prop, sigma))
    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()
    # save the figure

    plt.savefig(f"{out_dir}/{out_name}_acd.png", dpi=300)
    plt.show(block=False)
    # plt.pause(2)
    plt.close()


def plot_diffprop(f, out_dir='', out_name='diffprop_random_mp', fontsize=10):
    # Plot the line plot with error bars


    df = pd.read_csv(f)

    tot_props = [0, 0.2, 0.4, 0.6, 0.8]

    lloydL1_misc_avg, lloydL1_misc_err = df['lloydL1ians misc'], df['lloydL1ians misc err_bar']
    kmed_misc_avg, kmed_misc_err = df['lloydL1ians-L1 misc'], df['lloydL1ians-L1 misc err_bar']
    kmeans_misc_avg, kmeans_misc_err = df['kmeans misc'], df['kmeans misc err_bar']

    lloydL1_acd_avg, lloydL1_acd_err = df['lloydL1ians acd'], df['lloydL1ians acd err_bar']
    kmed_acd_avg, kmed_acd_err = df['lloydL1ians-L1 acd'], df['lloydL1ians-L1 acd err_bar']
    kmeans_acd_avg, kmeans_acd_err = df['kmeans acd'], df['kmeans acd err_bar']

    figsize = (8, 6)
    fig, ax = plt.subplots()

    plt.plot(tot_props, lloydL1_misc_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(tot_props, lloydL1_misc_avg, yerr=lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_props, kmed_misc_avg, '--', label='k-median', color="purple")
    plt.errorbar(tot_props, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_props, kmeans_misc_avg, '-', label='Llyod (k-means)', color="blue")
    plt.errorbar(tot_props, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(tot_props)

    plt.xlabel("Noise Proportion",fontdict={'fontsize':fontsize})
    plt.ylabel("MP",fontdict={'fontsize':fontsize})
    # plt.title(title+'_mp')
    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()

    # save the figure
    temp = time.time()
    temp = 0
    plt.savefig(f"{out_dir}/{out_name}_mp.png", dpi=300)

    plt.show(block=False)
    # plt.pause(2)

    # Plot the line plot with error bars
    # figsize = (8, 6)
    fig, ax = plt.subplots()

    plt.plot(tot_props, lloydL1_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(tot_props, lloydL1_acd_avg, yerr=lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_props, kmed_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(tot_props, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(tot_props, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(tot_props, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,lloydL1_acd_avg,kmed_acd_avg))+0.3)
    ax.set_xticks(tot_props)

    plt.xlabel("Noise Proportion", fontdict={'fontsize':fontsize})
    plt.ylabel("ACD", fontdict={'fontsize':fontsize})
    # plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_sigma_%g" % (num_centroids, radius, prop, sigma))
    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()
    # save the figure

    plt.savefig(f"{out_dir}/{out_name}_acd.png", dpi=300)
    plt.show(block=False)
    # plt.pause(2)
    plt.close()

def plot_diffvar(f, out_dir='', out_name='', fontsize=10):
    # Plot the line plot with error bars

    df = pd.read_csv(f)
    num_centroids = 4
    dim = 10
    sigma_out_vec = np.trunc(np.linspace(1, 20, 11))

    lloydL1_misc_avg, lloydL1_misc_err = df['lloydL1ians misc'], df['lloydL1ians misc err_bar']
    kmed_misc_avg, kmed_misc_err = df['lloydL1ians-L1 misc'], df['lloydL1ians-L1 misc err_bar']
    kmeans_misc_avg, kmeans_misc_err = df['kmeans misc'], df['kmeans misc err_bar']

    lloydL1_acd_avg, lloydL1_acd_err = df['lloydL1ians acd'], df['lloydL1ians acd err_bar']
    kmed_acd_avg, kmed_acd_err = df['lloydL1ians-L1 acd'], df['lloydL1ians-L1 acd err_bar']
    kmeans_acd_avg, kmeans_acd_err = df['kmeans acd'], df['kmeans acd err_bar']

    fig, ax = plt.subplots()

    plt.plot(sigma_out_vec, lloydL1_misc_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(sigma_out_vec, lloydL1_misc_avg, yerr=lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(sigma_out_vec, kmed_misc_avg, '--', label='k-median', color="purple")
    plt.errorbar(sigma_out_vec, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(sigma_out_vec, kmeans_misc_avg, '-', label='Llyod (k-means)', color="blue")
    plt.errorbar(sigma_out_vec, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(sigma_out_vec)

    plt.xlabel("Noise Standard Deviation",fontdict={'fontsize':fontsize})
    plt.ylabel("MP",fontdict={'fontsize':fontsize})
    # plt.title("Plot of misc_prop: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids, radius, prop, dim))

    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()
    # save the figure

    temp = time.time()
    temp = 0
    # plt.savefig(f'{out_dir}/misc_prop_%g_clusters_%f.png' % (num_centroids, temp), dpi=300)
    plt.savefig(f"{out_dir}/{out_name}_mp.png", dpi=300)
    plt.show(block=False)
    plt.pause(2)

    # Plot the line plot with error bars

    fig, ax = plt.subplots()

    plt.plot(sigma_out_vec, lloydL1_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(sigma_out_vec, lloydL1_acd_avg, yerr=lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(sigma_out_vec, kmed_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(sigma_out_vec, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(sigma_out_vec, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(sigma_out_vec, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,lloydL1_acd_avg,kmed_acd_avg))+0.3)
    ax.set_xticks(sigma_out_vec)

    plt.xlabel("Noise Standard Deviation",fontdict={'fontsize':fontsize})
    plt.ylabel("ACD",fontdict={'fontsize':fontsize})
    # plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids, radius, prop, dim))

    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()
    # save the figure

    # plt.savefig(f'{out_dir}/acd_%g_clusters_%f.png' % (num_centroids, temp), dpi=300)
    plt.savefig(f"{out_dir}/{out_name}_acd.png", dpi=300)

    plt.show(block=False)
    # plt.pause(2)
    plt.close()


def plot_diffrad(f, out_dir='', out_name='', fontsize=10):
    df = pd.read_csv(f)
    num_centroids = 4
    dim = 10
    rad_out_vec = np.trunc(np.linspace(0, 100, 11))

    lloydL1_misc_avg, lloydL1_misc_err = df['lloydL1ians misc'], df['lloydL1ians misc err_bar']
    kmed_misc_avg, kmed_misc_err = df['lloydL1ians-L1 misc'], df['lloydL1ians-L1 misc err_bar']
    kmeans_misc_avg, kmeans_misc_err = df['kmeans misc'], df['kmeans misc err_bar']

    lloydL1_acd_avg, lloydL1_acd_err = df['lloydL1ians acd'], df['lloydL1ians acd err_bar']
    kmed_acd_avg, kmed_acd_err = df['lloydL1ians-L1 acd'], df['lloydL1ians-L1 acd err_bar']
    kmeans_acd_avg, kmeans_acd_err = df['kmeans acd'], df['kmeans acd err_bar']

    # Plot the line plot with error bars

    # break

    fig, ax = plt.subplots()

    plt.plot(rad_out_vec, lloydL1_misc_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(rad_out_vec, lloydL1_misc_avg, yerr=lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmed_misc_avg, '--', label='k-median', color="purple")
    plt.errorbar(rad_out_vec, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmeans_misc_avg, '-', label='Llyod (k-means)',color="blue")
    plt.errorbar(rad_out_vec, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(rad_out_vec)

    plt.xlabel("Noise Radius")
    plt.ylabel("MP")
    # plt.title("Plot of misc_prop: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids, radius, prop, dim))

    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()
    # save the figure

    temp = time.time()
    temp = 0
    # plt.savefig(f'{out_dir}/misc_prop_%g_clusters_%f.png' % (num_centroids, temp), dpi=300)
    plt.savefig(f"{out_dir}/{out_name}_mp.png", dpi=300)
    plt.show(block=False)
    # plt.pause(2)

    # Plot the line plot with error bars

    fig, ax = plt.subplots()

    plt.plot(rad_out_vec, lloydL1_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(rad_out_vec, lloydL1_acd_avg, yerr=lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmed_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(rad_out_vec, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(rad_out_vec, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,lloydL1_acd_avg,kmed_acd_avg))+0.3)
    ax.set_xticks(rad_out_vec)

    plt.xlabel("Noise Radius")
    plt.ylabel("ACD")
    # plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids, radius, prop, dim))

    # Add a legend and show the plot
    plt.legend()
    plt.tight_layout()
    # save the figure

    # plt.savefig(f'{out_dir}/acd_%g_clusters_%f.png' % (num_centroids, temp), dpi=300)
    plt.savefig(f"{out_dir}/{out_name}_acd.png", dpi=300)
    plt.show(block=False)
    # plt.pause(2)
    plt.close()



if __name__ == '__main__':

    # in_dir = 'out-R_5000-S_100-20230516'
    # in_dir = 'R_5000-S_100-O_True-20230525'
    # in_dir = 'R_5000-S_100-O_True-20230610'
    in_dir = 'out/std_2/R_5000-S_100-O_True'
    out_dir = f'{in_dir}/paper_plot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    R = 5000 # 5000  # number of repeats
    S = 100

    for alg_method in ['diffdim', 'diffrad', 'diffvar', 'diffprop']: #['diffdim', 'diffrad', 'diffvar', 'diffprop']:
        for init_method in ['omniscient', 'random']:
            if init_method == 'random':
                py = f"main_clustering_{alg_method}_{init_method}_py"
            else:
                py = f"main_clustering_{alg_method}_py"
            # if alg_method=='diffrad':
            #     py = py + '-std_01'
            f = f'{in_dir}/{init_method}/{py}/data_4_clusters.csv'
            print(f)
            fontsize=12
            try:
                if alg_method == 'diffdim':
                    plot_diffdim(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}', fontsize=fontsize)
                elif alg_method == 'diffvar':
                    plot_diffvar(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}', fontsize=fontsize)
                elif alg_method == 'diffrad':
                    plot_diffrad(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}', fontsize=fontsize)
                elif alg_method == 'diffprop':
                    plot_diffprop(f, out_dir=out_dir, out_name=f'{alg_method}_{init_method}', fontsize=fontsize)
                else:
                    raise NotImplementedError()
            except Exception as e:
                traceback.print_exc()

    print('finished')

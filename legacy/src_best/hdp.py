import numpy as np


def hdp(data_hdp, beta_hdp):
    n_hdp = data_hdp.shape[0]

    m1_hdp = int(max([1, np.floor(n_hdp * beta_hdp)]))

    all_radii = []

    for x_hdp in range(n_hdp):
        dist_x_hdp = [np.linalg.norm(data_hdp[x_hdp] - point_hdp) for point_hdp in data_hdp]

        ind_of_n_beta_smallest_distance = np.argsort(dist_x_hdp)[m1_hdp - 1]

        all_radii.append(dist_x_hdp[ind_of_n_beta_smallest_distance])

    ind_hdp = np.argmin(all_radii)

    cent_temp = data_hdp[ind_hdp]

    return cent_temp

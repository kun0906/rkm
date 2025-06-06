import numpy as np
from init_2_cent import *


def iodk(dataset_temp, num_k, m1, m, beta):
    dim = dataset_temp.shape[1]

    data_size = dataset_temp.shape[0]

    mu_list = [np.nan + np.zeros(dim) for _ in range(num_k)]

    if data_size < m1 + m:

        totdist_temp = np.array(np.inf)

    else:

        if num_k == 2:

            mu_list[num_k - 1], mu_list[num_k - 2], totdist_temp = iod2(dataset_temp, m1, m, beta)

            totdist_temp = np.array(totdist_temp)

        else:

            n = dataset_temp.shape[0]

            mu_list[0] = hdp(dataset_temp, m1 / n)

            distances_mu1 = [np.linalg.norm(point - mu_list[0]) for point in dataset_temp]

            # Sort the distances and get the indices of the n*alpha/4 smallest distances
            indices_of_smallest_distances = np.argsort(distances_mu1)[:int(m1)]

            # Select the n * alpha/4 points with the smallest distances
            p_set = dataset_temp[indices_of_smallest_distances]

            distances_p_set = [np.linalg.norm(point - mu_list[0]) for point in p_set]

            dist_p_set = np.sort(distances_p_set)[int(np.floor(p_set.shape[0] * (1 - beta))) - 1]

            # Convert the NumPy arrays into lists of tuples
            points_list1 = [tuple(point) for point in dataset_temp]
            points_list2 = [tuple(point) for point in p_set]

            # Use list comprehensions to remove points from points_list1 that are in points_list2
            filtered_points_list = [point for point in points_list1 if point not in points_list2]

            # Convert the filtered list of tuples back to a NumPy array if needed
            pbar_set = np.array(filtered_points_list)

            pbar_set_size_loop = []

            pbar_set_size_loop.append(pbar_set.shape[0])

            mu_list_rest_temp0, dist_pbar_set = iodk(pbar_set, num_k - 1, m1, m, beta)

            mu_list_rest_temp = []

            totdist_loop = []

            mu_list_rest_temp.append(mu_list_rest_temp0)

            totdist_loop.append(np.array([dist_p_set + dist_pbar_set]))

            distances_mu0 = [np.linalg.norm(point - mu_list[0]) for point in dataset_temp]

            tot_count = int(np.ceil((n - m1) / m))

            for ell in range(tot_count - 2):

                # Sort the distances and get the indices of the n(1-beta) smallest distances
                ind_of_smallest_dist_loop = np.argsort(distances_mu0)[: m1 + (ell + 1) * m - 1]

                # Select the n(1-alpha/4) points with the smallest distances
                p_set_loop = dataset_temp[ind_of_smallest_dist_loop]

                distances_p_set_loop = [np.linalg.norm(point - mu_list[0]) for point in p_set_loop]

                dist_p_set_loop = np.sort(distances_p_set_loop)[int(np.floor(p_set_loop.shape[0] * (1 - beta))) - 1]

                # Convert the NumPy arrays into lists of tuples
                points_list1_loop = [tuple(point) for point in dataset_temp]
                points_list2_loop = [tuple(point) for point in p_set_loop]

                # Use list comprehensions to remove points from points_list1 that are in points_list2
                filtered_points_list_loop = [point for point in points_list1_loop if point not in points_list2_loop]

                # Convert the filtered list of tuples back to a NumPy array if needed
                pbar_set_loop = np.array(filtered_points_list_loop)

                pbar_set_size_loop.append(pbar_set_loop.shape[0])

                if pbar_set_loop.shape[0] <= m1:

                    temp_num_k = num_k - 1

                    mu_list_rest_temp_temp = [np.nan + np.zeros(dim) for _ in range(temp_num_k)]

                    mu_list_rest_temp.append(np.array(mu_list_rest_temp_temp))

                    totdist_loop.append(np.array(np.inf))

                else:

                    temp_num_k = num_k - 1

                    mu_list_rest_temp_temp, dist_pbar_set_loop = iodk(pbar_set_loop, temp_num_k, m1, m, beta)

                    mu_list_rest_temp.append(mu_list_rest_temp_temp)

                    totdist_loop.append(np.array([dist_p_set_loop + dist_pbar_set_loop]))

            try:

                totdist_loop = [float(value_temp) for value_temp in totdist_loop]

                ell_star = np.argmin(np.array(totdist_loop))

                mu_list_rest = mu_list_rest_temp[ell_star]

                mu_list[1:num_k] = mu_list_rest

                totdist_temp = np.array(totdist_loop[ell_star])

            except ValueError:

                print("\n totdist_temp \n")

                print(totdist_loop)

    return np.array(mu_list), np.array(totdist_temp)

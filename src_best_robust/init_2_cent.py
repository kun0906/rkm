import numpy as np
from hdp import hdp


def iod2(dataset, m1, m, beta):

    n = dataset.shape[0]

    dim = dataset.shape[1]

    if n < m1+m:

        mu1_star = np.zeros(dim)+np.nan

        mu2_star = np.zeros(dim)+np.nan

        totdist_temp = np.array(np.inf)

    else:

        mu1_temp = hdp(dataset, np.min([1, m1/n]))

        distances_mu1 = [np.linalg.norm(point-mu1_temp) for point in dataset]

        # Sort the distances and get the indices of the n*alpha/4 smallest distances
        indices_of_smallest_distances = np.argsort(distances_mu1)[:int(m1) - 1]

        # Select the n * alpha/4 points with the smallest distances
        p_set = dataset[indices_of_smallest_distances]

        distances_p_set = [np.linalg.norm(point - mu1_temp) for point in p_set]

        dist_p_set = np.sort(distances_p_set)[int(np.floor(p_set.shape[0]*(1-beta)))-1]

        # Convert the NumPy arrays into lists of tuples
        points_list1 = [tuple(point) for point in dataset]
        points_list2 = [tuple(point) for point in p_set]

        # Use list comprehensions to remove points from points_list1 that are in points_list2
        filtered_points_list = [point for point in points_list1 if point not in points_list2]

        # Convert the filtered list of tuples back to a NumPy array if needed
        pbar_set = np.array(filtered_points_list)

        pbar_set_size_loop = []

        pbar_set_size_loop.append(pbar_set.shape[0])

        mu2_temp = hdp(pbar_set, 1 - beta)

        distances_pbar_set = [np.linalg.norm(point - mu2_temp) for point in pbar_set]

        dist_pbar_set = np.sort(distances_pbar_set)[int(np.ceil(pbar_set.shape[0] * (1 - beta))) - 1]

        totdist_temp = [dist_p_set + dist_pbar_set]

        mu2_loop = [np.copy(mu2_temp)]

        totdist_loop = [np.copy(totdist_temp)]

        tot_count = int(np.ceil((n - m1) / m))

        for ell in range(tot_count-1):

            # Sort the distances and get the indices of the n(1-beta) smallest distances
            ind_of_smallest_dist_loop = np.argsort(distances_mu1)[: m1 + (ell+1) * m - 1]

            # Select the n(1-alpha/4) points with the smallest distances
            p_set_loop = dataset[ind_of_smallest_dist_loop]

            distances_p_set_loop = [np.linalg.norm(point - mu1_temp) for point in p_set_loop]

            dist_p_set_loop = np.sort(distances_p_set_loop)[int(np.floor(p_set_loop.shape[0] * (1 - beta))) - 1]

            # Convert the NumPy arrays into lists of tuples
            points_list1_loop = [tuple(point) for point in dataset]
            points_list2_loop = [tuple(point) for point in p_set_loop]

            # Use list comprehensions to remove points from points_list1 that are in points_list2
            filtered_points_list_loop = [point for point in points_list1_loop if point not in points_list2_loop]

            # Convert the filtered list of tuples back to a NumPy array if needed
            pbar_set_loop = np.array(filtered_points_list_loop)

            pbar_set_size_loop.append(pbar_set_loop.shape[0])

            if pbar_set_loop.shape[0] == 0:

                mu1_star = np.zeros(dim) + np.nan

                mu2_star = np.zeros(dim) + np.nan

                totdist_temp = np.array(np.inf)

            else:
                mu2_temp_loop = hdp(pbar_set_loop, 1 - beta)

                distances_pbar_set_loop = [np.linalg.norm(point - mu2_temp_loop) for point in pbar_set_loop]

                dist_pbar_set_loop = np.sort(distances_pbar_set_loop)[int(np.floor(pbar_set_loop.shape[0] * (1 - beta))) - 1]

                totdist_loop.append(np.array([dist_p_set_loop + dist_pbar_set_loop]))

                mu2_loop.append(mu2_temp_loop)

        ell_star = np.argmin(totdist_loop)

        mu1_star = mu1_temp

        mu2_star = mu2_loop[ell_star]

        totdist_temp = np.array(totdist_loop[ell_star])

    return mu1_star, mu2_star, totdist_temp


v0.7.4: Run robust_init on reduced data (after robust_lp)


v0.7.3: Add initialization inside of robust_lp on for random initialization


v0.7.2: Add  'robust_lp_k_medians_l2', 'robust_lp_k_medians_l1', 'robust_lp_k_means'


v0.7.1: correct mps in for in clustering_random.py


v0.7.0: fix bugs 
   1. remove n_init 
   2. add initial indices 


v0.6.9: add init_k_cent.py (robust initialization)
   

v0.6.8: add robust_sdp_lp.py
   Srivastava, Prateek R., Purnamrita Sarkar, and Grani A. Hanasusanto. "A robust spectral clustering algorithm
    for sub-Gaussian mixture models with outliers." Operations Research 71, no. 1 (2023): 224-244.


v0.6.7: sdp_lp_relaxation.py
   
   1. prob.solve(solver=cp.SCS, warm_start=True)
      very slow even for 400 points 
   2. prob.solve(solver=cp.CVXOPT, warm_start=True)
   # cvxopt==1.3.2   # verflowError: number of elements exceeds INT_MAX even for 400 points


v0.6.6: Add new spectral embedding 

1. New spectral embedding from: Löffler, M., Zhang, A. Y., & Zhou, H. H. (2021). Optimality of spectral clustering in the Gaussian mixture model.
    Annals of Statistics, 49(5), 2506–2530. https://doi.org/10.1214/20-AOS2044

2. Rename the old sc_projection() as sc_projection_sklearn()


v0.6.5: Use rbf for sc and rsc. 

1. Only using affinity='rbf' for sc and rsc will ignore the oultier impact, so we use rbf.
2. rsc needs to be verified on 'rbf'.
3. Not using partial outliers.


v0.6.4: fix bugs in omniscient initialization for sc and rsc. 

1. Only use normal data for omniscient initialization
2. Only use affinity='knn' for sc and rsc. 
3. Rsc won't work worth affinity='rbf' when we update Ag = A - Ac 


v0.6.3: update omniscient initialization for sc and rsc. 

1. Add tun parameters and n_init for random initialization
2. Add partial outliers


v0.6.2: Tune parameters for sc and rsc using src_best or src

1. Add tun parameters and n_init for random initialization


v0.6.1: sigma = 1.0 (not 2.0) for main_diff_rad.py

1. In the paper results, we use sigma=1.0 for main_diff_rad.py.
   For other cases (diff_var, prop, and dim), we use sigma = 2.0.
  # sigma = args.cluster_std
   sigma = 1.0
2. Update paper_plot.py 
3. 



v0.6.0: Refactor the entire library.

1. Merge random and omniscient together
2. Rename clustering name
3. Rewrite the main framework



v0.5.7: Add 'rbf' for RSC and remove zero eigenvectors from RSC.

v0.5.6: Set v0 to robust_spectral_clustering.py to for reproducibility.

1. Set v0 for RSC
   v0 = rng.rand(min(L.shape))   # avoid random initialization for eigsh()
   # solve the normal eigenvalue problem
   if self.laplacian == 0:
   h, H = eigsh(L, self.k, which='SM', v0=v0)     # eigsh can involve random initialization
   # solve the generalized eigenvalue problem
   elif self.laplacian == 1:
   h, H = eigsh(L, self.k, D, which='SM', v0=v0)    # tol=1e-6, add 1e-10 to the eigsh()

2. Set normalize=True.
   RSC(k=k, nn=15, theta=50, m=n_neighbours/10,laplacian=1, normalize=True, verbose=False, random_state=random_state)

v0.5.5: Add tol=1e-10 for robust_spectral_clustering.py

elif self.laplacian == 1:
h, H = eigsh(L, self.k, D, which='SM', tol=1e-10)    # add 1e-10 to the eigsh()

v0.5.4: Update hpc_sbatch.py and submit to HPC.

v0.5.3: Add Robust Spectral Clustering

v0.5.2: Choose bandwidth and update plots

1. Choose rbf bandwidth
2. Update plots
3. Update bugs

v0.5.1: Change rbf to knn for spectral clustering

1. Change rbf to knn for spectral clustering due to there are 0s for outliers when using rbf kernel.
   Causing affinity_matrix has 0 for whole rows (except for diagonal), and Graph is not connected. "Graph is not fully
   connected, spectral embedding may not work as expected."
2. Install threadpoolctl==3.1.0

v0.5.0: Add spectral clustering

1. Add spectral clustering for every experiment, e.g., sc_omniscient() and sc_random()
2. Add more information for running the experiments in Readme.md.

v0.4.1: Add synthetic datasets and try different setting for 'letter_recognition' and 'pen_digits'

1. Add synthetic datasets
2. For each dataset, add random outliers and special outliers
3. without_outliers in data_gen() can has different results due to the effect of rng.choice()
4. Add centroids and colors for plot_data().

v0.4.0: Add new datasets

1. Add new datasets
2. Implement diffprop for the new datasets
3. Add process_batch_real.py

v0.3.6: Update diffrad

1. Choose the random centroids on all data (including normal and outliers)
2. Fix outlier std to 10
3. Rerun all the cases.

v0.3.5: Rerun all the cases

1. Modify main_clustering_diffprop_random.py for random initialization.
2. Update proess_batch.py with more stds

v0.3.4: Change the std of normal data from 2 to 1 for all cases

1. Change the std of normal data from 2 to 1 for all cases
2. Add adjusted_text and plot_centroids
3. Add std into args for all cases

v0.3.3: Update the way to find the best centroids.

1. Update the way to find the best centroids by minimizing the misclustering proportion.
2. Replace 'missc' with 'misc' in 'data'.
3. Rerun everything and save centroids to disk.

v0.3.3: Reduce the standard deviations for different noise radius.

1. Reduce the std from 2 to 1 for normal clusters.

v0.3.2: Add different noise proportions

1. Add diffprop
2. Update paper_plot.py
3. Update R_5000.sh

v0.3.1: Rewrite bash by python

1. Rewrite the bash with python3
   limit the number of spawned processes
2. Add for loop in the bash
3. Generate results without outliers

v0.2.0: Rewrite the codes with seperated python scripts.

v0.1.4: Use ground-truth for omniscient initialization

1. Use ground-truth for omniscient initialization
2. Don't need to align centroids for omniscient case when we plot.
3. Add different noise variances in main_all.py (gaussians10_ds)

v0.1.3: Random sample data for an unit sphere (as true centroids)

1. Random sample n_clusters=2, 3 data points from an unit sphere (used as true centroids)
2. Add hypersphere
3. Add hyperball
4. Plotting takes too much time when n_clusters > 5 (because of align_centroids())

v0.1.2: Change r.randint() to r.choice(replace=False)

1. Change r.randint() to r.choice(replace=False)
2. Update 10gaussians_ds (adding different noise ratios)
3. Update collect_results2.py (only show random + kmeans++)
4. Remove X and y in seed_res.dat to save disk space.
5. Error: Pycharm Failed to transfer file. Could not close the output stream for file.
   There is no disk space in the remote server, please remove "files" in "out" directory to address this issue.

v0.1.1: Add Latex_tables

1. Add Latex_tables to convert the csv and print out the latex tables
2. Add 10gaussians and update the corresponding functions.
3. Add collect_results.sh and add --mem=40G to address the out-of-memory issue killed by the host.

v0.1.0: Add spectral clustering

1. reCompute E[ACD] instead of E[E[ACD]]  for plotting
   Using std error instead of std
2. Add multiprocessing in main_all.py
3. Add a new dataset: diff3_outliers

v0.0.9: Add multiprocessing and recompute E[ACD] for plotting

1. reCompute E[ACD] instead of E[E[ACD]]  for plotting
   Using std error instead of std
2. Add multiprocessing in main_all.py
3. Add a new dataset: diff3_outliers

v0.0.8: Compute E[E[ACD]] instead of E[ACD] for plotting

1. Compute E[E[ACD]] instead of E[ACD] for plotting
2. Update plot_misclustered_errors for E[E[ACD]]
3. Update the corresponding functions

v0.0.7: Implement K-Median_L1

1. Add a new dataset (diff2_outliers)
2. Implement K-Median_L1
3. Update the initialization

v0.0.6: Add a new dataset and centroid_diff as an evaluation metric

1. Add a new dataset (constructed_3gaussians)
2. Add 'centroid_diff' = diff(centroid_pred, centroid_true).
3. Update the plots

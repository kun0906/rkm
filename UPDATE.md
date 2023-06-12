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

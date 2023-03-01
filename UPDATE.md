V0.0.9: Add multiprocessing and recompute E[ACD] for plotting

1. reCompute E[ACD] instead of E[E[ACD]]  for plotting
   Using std error instead of std
2. Add multiprocessing in main_all.py 
3. Add a new dataset: diff3_outliers


V0.0.8: Compute E[E[ACD]] instead of E[ACD] for plotting

1. Compute E[E[ACD]] instead of E[ACD] for plotting
2. Update plot_misclustered_errors for E[E[ACD]]
3. Update the corresponding functions 


V0.0.7: Implement K-Median_L1

1. Add a new dataset (diff2_outliers)
2. Implement K-Median_L1
3. Update the initialization 


V0.0.6: Add a new dataset and centroid_diff as an evaluation metric

1. Add a new dataset (constructed_3gaussians)
2. Add 'centroid_diff' = diff(centroid_pred, centroid_true).
3. Update the plots

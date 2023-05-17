import numpy as np

# true centroids
true = [[ 4.85743972, 1.18544478],
 [ 0.25856916, -4.99330972],
 [-3.0798247,  -3.93886783],
 [-0.65414847, -4.95702429],
 [-0.45717693, -4.97905506]]
true = np.asarray(true)

# the centroids obtained by kmeans when we use the true centroids as the initial centroids
before = [[ 5.40981993,  1.3431326 ],
 [-0.20658956, -7.36436882],
 [-0.11072462, 15.6398994 ],
 [-5.50119679, -3.15306737],
 [-0.67575555, -4.13573729]]
before = np.asarray(before)

# the centroids after alignment. Noticed that the centroids are different with 'before' (however, they have the minimal distance to true centroids)
after=[[-0.11072462, 15.6398994 ],
 [ 5.40981993,  1.3431326 ],
 [-5.50119679, -3.15306737],
 [-0.67575555, -4.13573729],
 [-0.20658956, -7.36436882]]
after = np.asarray(after)


d1 = np.sum(np.sum(np.square(before - true), axis=1), axis=0)
d2 = np.sum(np.sum(np.square(after - true), axis=1), axis=0)

print(d1, d2)




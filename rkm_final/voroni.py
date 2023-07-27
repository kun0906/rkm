import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Generate random data for two clusters
n_samples = 1000
mean1 = [8, -1]
mean2 = [-8, 1]

# mean1 = [-5, 0]
# mean2 = [5, 0]

cov = [[20, 0], [0, 20]]


a = mean1[0]
b = mean2[1]


# Centers
x1 = mean1[0]
y1 = mean1[1]

x2 = mean2[0]
y2 = mean2[1]



def y_L1(y): #For a>0,b>0 # Voronoi graph for L1

    ret = np.zeros_like(y)
    for i in range(len(y)):
        if y[i] > b:
            ret[i] = b
        elif y[i] < -b:
            ret[i] = -b
        else:
            ret[i] = y[i]
    return ret
#
# def dist(p1, p2, name='l1'):
#
#     return sum(abs(v1-v2 for v1, v2 in zip(p1, p2)))
# def y_L1(X, Y):
#
#     ret = [0] * len(X)
#     for i, (x, y) in enumerate(zip(X, Y)):
#
#         d1 = dist((x,y), mean1, 'l1')
#         d2 = dist((x,y), mean2, 'l1')
#         if d1 > d2:
#             ret[i] =
#         elif d1 < d2:
#             ret[i] =
#         else:
#             ret[i] =
#
#     return


# Generate a range of x values to plot
x = np.linspace(-20, 20, 1000)
y = np.linspace(-20, 20, 1000)


#Voronoi graph for L2
def y_L2(x):
    return (y1 + y2) / 2 + (-(x2 - x1) / (y2 - y1)) * (x - (x1 + x2) / 2)
    # return 1



rep = 100

vL1 = np.zeros(rep)
vL2 = np.zeros(rep)

for j in range(rep):
    random.seed(42)

    cluster1 = np.random.multivariate_normal(mean1, cov, n_samples)

    random.seed(40)
    cluster2 = np.random.multivariate_normal(mean2, cov, n_samples)

    # Define the point and the angle of rotation


    # Calculate the new coordinates after rotation
    c1_rot = np.copy(cluster1)
    c2_rot = np.copy(cluster2)


    c1_rot[:,0] = cluster1[:,0]
    c1_rot[:,1] = cluster1[:,1]

    c2_rot[:,0] = cluster2[:,0]
    c2_rot[:,1] = cluster2[:,1]

    # vL1[j] = (np.sum(y_L1(c2_rot[:,0])-c2_rot[:,1]<0)+np.sum(y_L1(c1_rot[:,0])-c1_rot[:,1]>0))/(2*len(c2_rot[:,0]))
    vL1[j] = (np.sum(y_L1(c2_rot[:, 1]) - c2_rot[:, 0] < 0) + np.sum(y_L1(c1_rot[:, 1]) - c1_rot[:, 0] > 0)) / (
                2 * len(c2_rot[:, 1]))

    vL2[j] = (np.sum(y_L2(c2_rot[:, 0]) - c2_rot[:, 1] < 0) + np.sum(y_L2(c1_rot[:, 0]) - c1_rot[:, 1] > 0)) / (
                2 * len(c2_rot[:, 0]))

print(np.mean(vL1),np.std(vL1))

print(np.mean(vL2),np.std(vL2))

plt.figure(figsize=(6, 6))

plt.scatter(c1_rot[:, 0], c1_rot[:, 1], marker='x', label = 'mean=(5,-1)')
plt.scatter(c2_rot[:, 0], c2_rot[:, 1], marker='o', facecolors='none', edgecolor='orange', label = 'mean=(-5,1)')

plt.plot(y_L1(y), y, 'red', label = 'Cluster border')
# plt.plot(x, y_L1(x), 'red', label = 'Cluster border')

plt.title('Cluster formation using $\ell_1$ metric')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-max(x), max(x))
plt.ylim(-max(x), max(x))

plt.legend()
plt.savefig("l1-clustering")
plt.show()

plt.clf()

plt.figure(figsize=(6, 6))

plt.scatter(c1_rot[:, 0], c1_rot[:, 1], marker='x', label = 'mean=(5,-1)')
plt.scatter(c2_rot[:, 0], c2_rot[:, 1], marker='o', facecolors='none', edgecolor='orange', label = 'mean=(-5,1)')

plt.plot(x, y_L2(x), 'black', label = 'Cluster border')

plt.title('Cluster formation using $\ell_2$ metric')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-max(x), max(x))
plt.ylim(-max(x), max(x))
plt.legend()
plt.savefig("l2-clustering")
plt.show()
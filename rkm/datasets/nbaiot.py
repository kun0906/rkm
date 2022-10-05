"""
NBAIOT
Data Set Information:

(a) Attribute being predicted:
-- Originally we aimed at distinguishing between benign and Malicious traffic data by means of anomaly detection techniques.
-- However, as the malicious data can be divided into 10 attacks carried by 2 botnets, the dataset can also be used for multi-class classification: 10 classes of attacks, plus 1 class of 'benign'.

(b) The study's results:
-- For each of the 9 IoT devices we trained and optimized a deep autoencoder on 2/3 of its benign data (i.e., the training set of each device). This was done to capture normal network traffic patterns.
-- The test data of each device comprised of the remaining 1/3 of benign data plus all the malicious data. On each test set we applied the respective trained (deep) autoencoder as an anomaly detector. The detection of anomalies (i.e., the cyberattacks launched from each of the above IoT devices) concluded with 100% TPR.


https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT#

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split


def nbaiot_diff_outliers(args, random_state=42, **kwargs):
	"""
	# two clusters ((-3,0), (3, 0)) with same covariance matrix and size in R^2, i.e., n1 = n2 = 500
	# 10% of outliers, i.e., (n1+n2)*0.1.
	# Move the outliers away from n1 over time.
	# e.g., 'r:0.1|mu:-10,0|cov:0.1,0.1', which means we draw 10% outliers from a Gaussian with mu=(-10, 0) and cov=(0.1, 0.1)

	params['p1'] == 'diff_outliers':
	Parameters
	----------
	params
	random_state

	Returns
	-------

	"""
	# r:0.1|mu:-3,0|cov:0.1,0.1|diff_outliers
	dataset_detail = args['DATASET']['detail']
	tmp = dataset_detail.split('|')
	ratio = float(tmp[0].split(':')[1])

	mu = tmp[1].split(':')[1].split(',')
	# mu = np.asarray([float(mu[0]), float(mu[1])])
	mu = float(mu[0])

	cov = tmp[2].split(':')[1].split(',')
	# cov = np.asarray([[float(cov[0]), 0], [0, float(cov[1])]])
	cov = float(cov[0])

	in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
	out_file = 'Xy.dat'
	out_file = os.path.join(in_dir, out_file)

	r = np.random.RandomState(random_state)

	def get_xy(ratio, mu, cov):

		f_normal = os.path.join(in_dir, 'benign_traffic.csv')
		X1 = pd.read_csv(f_normal).values
		y1 = np.asarray([0] * X1.shape[0])
		f_abnormal = os.path.join(in_dir, os.path.join('gafgyt_attacks', "tcp.csv"))
		X2 = pd.read_csv(f_abnormal).values
		y2 = np.asarray([1] * X2.shape[0])

		############
		# cluster 1
		n1 = 500
		X1, X1_test, y1, y1_test = sklearn.model_selection.train_test_split(X1, y1, train_size=n1, random_state=random_state,
		                                                        shuffle=True)

		############
		# cluster 2
		n2 = 500
		X2, _, y2, _ = sklearn.model_selection.train_test_split(X2, y2, train_size=n2, random_state=random_state,
		                                                        shuffle=True)

		# obtain ground truth centroids
		true_centroids = np.zeros((2, X1.shape[1]))
		true_centroids[0] = np.mean(X1, axis=0)
		true_centroids[1] = np.mean(X2, axis=0)

		# obtain initial centroids, i.e., random select 2 data points from cluster 1
		indices = r.choice(range(0, n1), size=2, replace=False)  # without replacement and random
		init_centroids = X1[indices]

		############
		# outliers
		n_outliers = int((n1 + n2) * ratio)
		mu = np.asarray([mu] + [0] * (X1.shape[1]-1))
		cov = [cov] + [0] * (X1.shape[1]-1)
		cov = np.diag(np.array(cov))
		X_outliers = r.multivariate_normal(mu, cov, size=n_outliers)
		# X_outliers, X1_test, _, _ = sklearn.model_selection.train_test_split(X1_test, y1_test, train_size=n_outliers,
		#                                                                     random_state=random_state,
		#                                                                     shuffle=True)
		y_outliers = np.asarray([2] * n_outliers)

		# Combine them togather
		X = np.concatenate([X1, X2, X_outliers], axis=0)
		y = np.concatenate([y1, y2, y_outliers], axis=0)

		delta_X = abs(mu[0])
		return X, y, true_centroids, init_centroids, delta_X

	X, y, true_centroids, init_centroids, delta_X = get_xy(ratio, mu, cov)

	is_show = args['IS_SHOW']
	if is_show:
		# Plot init seeds along side sample data
		fig, ax = plt.subplots()
		# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
		colors = ["r", "g", "b", "m", 'black']
		ax.scatter(X[:, 0], X[:, 1], c=y, marker="x", s=10, alpha=0.3, label='$G_1$')
		p = np.mean(X, axis=0)
		ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		offset = 0.3
		# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		xytext = (p[0] - offset, p[1] - offset)
		# print(xytext)
		ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
		            ha='center', va='center',  # textcoords='offset points',
		            bbox=dict(facecolor='none', edgecolor='b', pad=1),
		            arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
		                            connectionstyle="angle3, angleA=90,angleB=0"))

		ax.axvline(x=0, color='k', linestyle='--')
		ax.axhline(y=0, color='k', linestyle='--')
		ax.legend(loc='upper right', fontsize=13)
		if args['SHOW_TITLE']:
			plt.title(dataset_detail.replace(':', '\n'))

		if 'xlim' in kwargs:
			plt.xlim(kwargs['xlim'])
		else:
			plt.xlim([-6, 6])
		if 'ylim' in kwargs:
			plt.ylim(kwargs['ylim'])
		else:
			plt.ylim([-6, 6])

		fontsize = 13
		plt.xticks(fontsize=fontsize)
		plt.yticks(fontsize=fontsize)

		plt.tight_layout()
		# if not os.path.exists(params['OUT_DIR']):
		#     os.makedirs(params['OUT_DIR'])
		# f = os.path.join(args['OUT_DIR'], dataset_detail+'.png')
		f = args['data_file'] + '.png'
		print(f)
		plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()

	return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}


def nbaiot_mixed_clusters(args, random_state=42, **kwargs):
	"""
	# two clusters ((-3,0), (3, 0)) with same covariance matrix and size in R^2, i.e., n1 = n2 = 500
	# mix the two clusters with different ratio
	# e.g., 'r:0.4|mixed_clusters', which means we draw 40% data from cluster 1 and add them to cluster2, and vice versa for cluster 2.

	Parameters
	----------
	params
	random_state

	Returns
	-------

	"""
	# d:2|r:0.1|mixed_clusters
	dataset_detail = args['DATASET']['detail']
	tmp = dataset_detail.split('|')

	d = float(tmp[0].split(':')[1])

	ratio = float(tmp[1].split(':')[1])

	in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
	out_file = 'Xy.dat'
	out_file = os.path.join(in_dir, out_file)

	r = np.random.RandomState(random_state)

	def get_xy(d, ratio):
		f_normal = os.path.join(in_dir, 'benign_traffic.csv')
		X1 = pd.read_csv(f_normal).values
		y1 = np.asarray([0] * X1.shape[0])
		f_abnormal = os.path.join(in_dir, os.path.join('gafgyt_attacks', "tcp.csv"))
		X2 = pd.read_csv(f_abnormal).values
		y2 = np.asarray([1] * X2.shape[0])

		############
		# cluster 1
		n1 = 500
		X1, _, y1, _ = sklearn.model_selection.train_test_split(X1, y1, train_size=n1, random_state=random_state,
		                                                        shuffle=True)
		X1[:, 0] = X1[:, 0] - d

		############
		# cluster 2
		n2 = 500
		X2, _, y2, _ = sklearn.model_selection.train_test_split(X2, y2, train_size=n2, random_state=random_state,
		                                                        shuffle=True)
		X2[:, 0] = X2[:, 0] + d

		# obtain ground truth centroids
		true_centroids = np.zeros((2, X1.shape[1]))
		true_centroids[0] = np.mean(X1, axis=0)
		true_centroids[1] = np.mean(X2, axis=0)

		X = np.concatenate([X1, X2], axis=0)
		y = np.concatenate([y1, y2], axis=0)

		# obtain initial centroids after mixing the data
		X11, X12, y11, y12 = train_test_split(X1, y1, test_size=ratio, shuffle=True, random_state=random_state)
		X21, X22, y21, y22 = train_test_split(X2, y2, test_size=ratio, shuffle=True, random_state=random_state)
		# Mix them togather
		X1_ = np.concatenate([X11, X22], axis=0)
		# y1_ = np.concatenate([y11, y22], axis=0)
		X2_ = np.concatenate([X21, X12], axis=0)
		# y2_ = np.concatenate([y21, y12], axis=0)

		init_centroids = np.zeros((2, X1.shape[1]))
		init_centroids[0] = np.mean(X1_, axis=0)
		init_centroids[1] = np.mean(X2_, axis=0)

		delta_X = 2 * d

		return X, y, true_centroids, init_centroids, delta_X

	X, y, true_centroids, init_centroids, delta_X = get_xy(d, ratio)

	return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}


def stats(in_dir='datasets/NBAIOT/Danmini_Doorbell'):
	csvs = ['benign_traffic.csv'] + \
	       [os.path.join('gafgyt_attacks', f) for f in sorted(os.listdir(os.path.join(in_dir, 'gafgyt_attacks')))] + \
	       [os.path.join('mirai_attacks', f) for f in sorted(os.listdir(os.path.join(in_dir, 'mirai_attacks')))]
	for csv in csvs:
		f = os.path.join(in_dir, csv)
		df = pd.read_csv(f)
		print(df.shape, f)


if __name__ == '__main__':
	stats()
# nbaiot_diff_sigma_n({'N_CLIENTS': 0, 'N_CLUSTERS': 2, 'IS_PCA':True, 'DATASET': {'detail': 'n1_100+n2_100+n3_100:ratio_0.00:diff_sigma_n'}})

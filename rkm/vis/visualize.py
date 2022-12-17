"""

"""
# Email: kun.bj@outllok.com

import matplotlib.pyplot as plt
import numpy as np

precision = 3


def plot_misclustered_errors(resutls, out_file='.png', title='', is_show=True):
	fig, axes = plt.subplots()  # (width, height)
	# fig.suptitle(title  + ', centroids update')
	colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']

	for i_alg, alg_name in enumerate(resutls.keys()):
		X = []
		Y = []
		Y_errs = []
		for data_name, diff_outliers in resutls[alg_name].items():
			n_training_iterations = []
			for repeat_vs in diff_outliers:
				_tmp = [repeat['delta_X'] for repeat in repeat_vs]  # for different repeats, the delta is the same.
				X.append(_tmp[0])
				_tmp = [repeat['misclustered_error'] for repeat in repeat_vs]
				mu = float(f'{np.mean(_tmp):.{precision}f}')
				std = float(f'{np.std(_tmp):.{precision}f}')
				Y.append(mu)
				Y_errs.append(std)

				_tmp = [repeat['n_training_iterations'] for repeat in repeat_vs]
				_n_training_iterations = (
					float(f'{np.mean(_tmp):.{precision}f}'), float(f'{np.std(_tmp):.{precision}f}'))
				n_training_iterations.append(_n_training_iterations)
		print(f'{alg_name}: X: {X}, Y:{Y}, Y_errs: {Y_errs}, n_training_iterations: {n_training_iterations}')
		# plt.errorbar(X, Y, Y_errs)

		if alg_name == 'kmeans':
			label = 'K-Means'
			color = 'blue'
			ecolor = 'tab:red'
		elif alg_name == 'kmedian':
			label = 'K-Median'
			color = 'green'
			ecolor = 'tab:brown'
		else:
			label = 'K-Median(Tukey)'
			color = 'm'
			ecolor = 'tab:cyan'
		plt.errorbar(X, Y, Y_errs, fmt='*-',
		             capsize=3, color=color, ecolor=ecolor,
		             markersize=8, markerfacecolor='black',
		             label=label)

	font_size = 15
	plt.legend(loc='upper right', fontsize=font_size - 2)  # bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
	axes.set_ylabel('Average Misclustered Error: $A_T$')
	axes.set_xlabel('$\Delta$')  # the distance between outlier and origin.
	X = [_v for _i, _v in enumerate(X) if _i != 1]
	axes.set_xticks(X)

	plt.tight_layout()
	with open(out_file, 'wb') as f:
		plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		plt.show()
	# plt.clf()
	plt.close(fig)


def plot_mixted_clusters(resutls, out_file='.png', n_th = 5, title='', is_show=True):
	fig, axes = plt.subplots()  # (width, height)
	# fig.suptitle(title  + ', centroids update')
	colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']

	for i_alg, alg_name in enumerate(resutls.keys()):
		X = []
		Y = []
		Y_errs = []
		for data_name, diff_d in resutls[alg_name].items():
			n_training_iterations = []
			for repeat_vs in diff_d:
				_tmp = [repeat['delta_X'] for repeat in repeat_vs]  # for different repeats, the delta is the same.
				X.append(_tmp[0])
				# _tmp = [repeat['misclustered_error'] for repeat in repeat_vs]    # when training is finished, we get the misclustered error
				_tmp = [repeat['history'][n_th]['scores']['misclustered_error'] for repeat in
				        repeat_vs]  # during the training, we get the misclustered error after 5 iterations.
				mu = float(f'{np.mean(_tmp):.{precision}f}')
				std = float(f'{np.std(_tmp):.{precision}f}')
				Y.append(mu)
				Y_errs.append(std)

				_tmp = [repeat['n_training_iterations'] for repeat in repeat_vs]
				_n_training_iterations = (
					float(f'{np.mean(_tmp):.{precision}f}'), float(f'{np.std(_tmp):.{precision}f}'))
				n_training_iterations.append(_n_training_iterations)
		print(f'{alg_name}: X: {X}, Y:{Y}, Y_errs: {Y_errs}, n_training_iterations: {n_training_iterations}')
		# plt.errorbar(X, Y, Y_errs)

		if alg_name == 'kmeans':
			label = 'K-Means'
			color = 'blue'
			ecolor = 'tab:red'
		elif alg_name =='kmedian':
			label = 'K-Median'
			color = 'green'
			ecolor = 'tab:brown'
		else:
			label = 'K-Median(Tukey)'
			color = 'm'
			ecolor = 'tab:cyan'
		plt.errorbar(X, Y, Y_errs, fmt='*-',
		             capsize=3, color=color, ecolor=ecolor,
		             markersize=8, markerfacecolor='black',
		             label=label)

	font_size = 15
	plt.legend(loc='upper right', fontsize=font_size - 2)  # bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
	axes.set_ylabel(f'Average Misclustered Error: $A_{n_th}$')
	axes.set_xlabel('$\Delta$')  # the distance between outlier and origin.
	X = [_v for _i, _v in enumerate(X) if _i != 1]
	axes.set_xticks(X)

	plt.tight_layout()
	with open(out_file, 'wb') as f:
		plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		plt.show()
	# plt.clf()
	plt.close(fig)


def plot_centroids(X, y_pred, centroids, is_show=True):
	fig, axes = plt.subplots()  # (width, height)
	plt.scatter(X[:, 0], X[:, 1], c=y_pred)
	plt.axvline(x=0)
	plt.axhline(y=0)

	for i in range(centroids.shape[0]):
		c = centroids[i]
		plt.scatter(c[0], c[1], marker='x', color='red')
		offset = 0.
		xytext = (c[0], c[1] + offset)
		axes.annotate(f'({c[0]:.1f}, {c[1]:.1f})', xy=(c[0], c[1]), xytext=xytext, fontsize=15, color='black',
		              ha='center', va='center',  # textcoords='offset points',
		              bbox=dict(facecolor='none', edgecolor='gray', pad=1),
		              arrowprops=dict(arrowstyle="->", color='gray', shrinkA=1,
		                              connectionstyle="angle3, angleA=90,angleB=0"))

	plt.tight_layout()
	if is_show: plt.show()
	# plt.clf()
	plt.close(fig)

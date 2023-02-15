"""

"""
# Email: kun.bj@outllok.com
import os.path

import matplotlib.pyplot as plt
import numpy as np

from rkm.utils.common import fmt_np

precision = 3


def plot_misclustered_errors(resutls, out_file='.png', title='', error_method = 'misclustered_error',
                             is_show=True, raw_n_th=5, verbose=10, case = None, init_method='random'):
	fig, axes = plt.subplots()  #  figsize=(8, 6) (width, height)
	fig2, axes2 = plt.subplots(nrows=4, ncols=7, sharex=False,
	                         sharey=False, figsize=(15, 13))  # figsize=(8, 6) (width, height)
	# fig.suptitle(title  + ', centroids update')
	colors = ['blue', 'green', 'm', 'b', 'r', 'tab:brown', 'tab:green', 'orange', ]
	if raw_n_th==-1:
		raw_n_th = np.inf
	markers_fmt = ['*-', 'o-', '^-', 'V-']
	table_res = []
	py_names = [
		'kmeans',
		'kmedian_l1',
		'kmedian',  # our method
		# 'kmedian_tukey',
	]
	for i_alg, alg_name in enumerate(py_names): # enumerate(resutls.keys()): algorithm
		X = []
		Y = []
		Y_errs = []
		for data_name, diff_outliers in resutls[alg_name].items(): # coviance matrix
			n_training_iterations = []
			for i_delta, delta_vs in enumerate(diff_outliers): # covirance matrix
				_tmp = [repeat['delta_X'] for repeat in delta_vs]  # for different repeats, the delta is the same.
				X.append(_tmp[0])
				# # show the errors at the n_th iteration
				# n_th = min(raw_n_th, min(repeat['n_training_iterations'] for repeat in repeat_vs))   # index starts from 0 # index starts from 0)   # index starts from 0
				# _tmp = [repeat['history'][n_th-1]['scores'][error_method] for repeat in repeat_vs]
				_tmp = []
				n_ths = []
				p = delta_vs[0]['delta_X']
				acd_metrics = []
				for r_idx, repeat in enumerate(delta_vs): # repeatations
					# if raw_n_th > the total iterations, then use the final results.
					n_th = min(raw_n_th, repeat['n_training_iterations'])
					_tmp.append(repeat['history'][n_th - 1]['scores'][error_method])    # ACD
					n_ths.append(n_th)
					if verbose >=10 and raw_n_th == np.inf:
						print(f'{r_idx}th final centroids:', repeat['history'][- 1]['centroids'], ' initial:', repeat['history'][0]['centroids'])
				if verbose >= 10:
					print('p:',p, ' iterations: ', n_th, n_ths, raw_n_th)
				acd_metrics.extend(_tmp)

				mu = float(f'{np.mean(_tmp):.{precision}f}')
				std = float(f'{np.std(_tmp):.{precision}f}')
				Y.append(mu)
				Y_errs.append(std)

				_tmp = [repeat['n_training_iterations'] for repeat in delta_vs]
				_n_training_iterations = (
					float(f'{np.mean(_tmp):.{precision}f}'), float(f'{np.std(_tmp):.{precision}f}'))
				n_training_iterations.append(_n_training_iterations)

				if i_delta < 7:
					# plot all in one
					# p = delta_vs[0]['delta_X']
					# axes2[0, i_repeat].hist(acd_metrics, color=colors[i_alg], label=f'{alg_name}: {data_name}: {p}')
					axes2[0, i_delta].hist(acd_metrics, color=colors[i_alg], label=f'{alg_name}:{mu}+/-{std}', alpha=0.5, density=False)
					# if i_delta == 0:
					axes2[0, i_delta].legend(loc='upper right', fontsize=6)
					axes2[0, i_delta].set_title(f'X_noise:{p}')
					axes2[0, i_delta].set_xlabel('ACD')
					axes2[0, i_delta].set_ylabel('Frequency')

					# plot each algorithm's result
					# p = delta_vs[0]['delta_X']
					# axes2[i_repeat].hist(acd_metrics, color=colors[i_alg], label=f'{alg_name}: {data_name}: {p}')
					axes2[i_alg+1, i_delta].hist(acd_metrics, color=colors[i_alg], label=f'{alg_name}:{mu}+/-{std}', alpha=0.5, density=False)
					# if i_delta == 0:
					axes2[i_alg+1, i_delta].legend(loc='upper right', fontsize=6)
					axes2[i_alg+1,i_delta].set_title(f'X_noise:{p}')
					axes2[i_alg+1, i_delta].set_xlabel(f'ACD') #:{mu}+/-{std}
					axes2[i_alg+1, i_delta].set_ylabel('Frequency')

				# print(f'{alg_name}: X: {X}, Y:{Y}, Y_errs: {Y_errs}, n_training_iterations: {n_training_iterations}')
		vs = [f'{_mu:.3f}+/-{_std:.3f}' for _mu, _std in zip(Y, Y_errs)]
		print(f'{alg_name}: X: {X}, Y:{vs}, n_training_iterations: {n_training_iterations}')
		# plt.errorbar(X, Y, Y_errs)
		table_res.append(alg_name +' & ' + ' & '.join(vs))
		if alg_name == 'kmeans':
			if raw_n_th == np.inf: # final result
				label = f'K-Means'
			else:
				# label = f'K-Means({init_method}): (iters){np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
				label = f'K-Means({init_method},p_{p}):{mu:.2f}+/-{std:.2f}'
			color = 'blue'
			ecolor = 'tab:red'
		elif alg_name =='kmedian':
			if raw_n_th == np.inf:
				label = f'K-Median'
			else:
				# label = f'K-Median({init_method}): {np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
				label = f'K-Median({init_method},p_{p}):{mu:.2f}+/-{std:.2f}'
			color = 'green'
			ecolor = 'tab:brown'
		elif alg_name =='kmedian_l1':
			if raw_n_th == np.inf:
				label = f'K-Median_L1'
			else:
				# label = f'K-Median_L1({init_method}): {np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
				label = f'K-Median_L1({init_method},p_{p}):{mu:.2f}+/-{std:.2f}'
			color = 'm'
			ecolor = 'tab:cyan'
		else:
			label = 'K-Median(Tukey)'
			# color = 'm'
			# ecolor = 'tab:cyan'
		axes.errorbar(X, Y, Y_errs, fmt=markers_fmt[i_alg],
		             capsize=3, color=color, ecolor=ecolor,
		             markersize=8, markerfacecolor='black',
		             label=label, alpha=1)

	print('\n'.join(table_res))
	font_size = 15
	axes.legend(loc='upper left', fontsize=font_size - 2)  # bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
	# if error_method in ['centroid_diff', 'centroid_diff2']:
	# 	ylabel = '$\\left||\mu-\mu^{*}\\right||$' + f'$ACD_{n_th-1}$'
	# else:
	# 	ylabel = 'Average Misclassified Error (AME)'
	if error_method in ['centroid_diff', 'centroid_diff2']:
		ylabel = 'Diff' if error_method == 'centroid_diff' else 'Diff2'
		# ylabel = f'{ylabel}: ' + '$\\left||\mu-\mu^{*}\\right||$' + ': $ACD_{' + f'{n_th-1}' + '}$'
		# ylabel = '$ACD_{' + f'{n_th}' + 'th}$'   # start from index 1
		if raw_n_th==np.inf:
			ylabel = '$ACD^{*}$'  # converged results
		else:
			ylabel = '$ACD_{' + f'{n_th}' + 'th}$'  # start from index 1

		xlabel = '$x_{noise}$'
	else:
		ylabel = 'Average Misclustered Error' + ': $ACD_{' + f'{n_th}' + '}$'   # start from index 1
		xlabel = '$x_{noise}$'
	axes.set_ylabel(f'{ylabel}', fontsize=font_size)
	axes.set_xlabel(xlabel, fontsize=font_size)  # the distance between outlier and origin.
	# X = [_v for _i, _v in enumerate(X) if _i != 1]
	axes.set_xticks(X)
	axes.tick_params(axis='both', which='major', labelsize=font_size-2)
	# axes.xticks(fontsize=font_size-2)

	fig.tight_layout()
	with open(out_file, 'wb') as f:
		fig.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		fig.show()
	# plt.clf()
	plt.close(fig)

	if raw_n_th == -1 or raw_n_th == np.inf:
		title = '$ACD_{' + '*' + '}$'
	else:
		title = '$ACD_{' + f'{raw_n_th}' + '}$'
	fig2.suptitle(title)
	fig2.tight_layout()
	with open(out_file[:-4]+'_hist.png', 'wb') as f:
		fig2.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		fig2.show()
	plt.close(fig2)


def plot_mixed_clusters(resutls, out_file='.png', raw_n_th = 5, title='', error_method='misclustered_error',
                        is_show=True, verbose=0, case = '', init_method='random'):
	fig, axes = plt.subplots()  # (width, height)
	# fig.suptitle(title  + ', centroids update')
	colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']
	if raw_n_th == -1:
		raw_n_th = np.inf
	table_res = []
	markers_fmt = ['*-', 'o-', '^-', 'V-']
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
				# show the errors at the n_th iteration
				# n_th = min(raw_n_th, min(repeat['n_training_iterations'] for repeat in repeat_vs))   # index starts from 0
				# print(n_th, repeat_vs[0]['n_training_iterations'], len(repeat_vs[0]['history']), flush=True, )
				# _tmp = [repeat['history'][n_th-1]['scores'][error_method] for repeat in
				#         repeat_vs]  # during the training, we get the misclustered error after 5 iterations.
				_tmp = []
				n_ths = []
				p = repeat_vs[0]['delta_X']
				for r_idx, repeat in enumerate(repeat_vs):
					# if raw_n_th > the total iterations, then use the final results.
					n_th = min(raw_n_th, repeat['n_training_iterations'])
					_tmp.append(repeat['history'][n_th - 1]['scores'][error_method])
					n_ths.append(n_th)
					if raw_n_th == np.inf:
						print(f'{r_idx}th final centroids:', repeat['history'][- 1]['centroids'], ' initial:', repeat['history'][0]['centroids'])
				if verbose>=10: print('p:', p, ' iterations: ', n_th, n_ths, raw_n_th)
				mu = float(f'{np.mean(_tmp):.{precision}f}')
				std = float(f'{np.std(_tmp):.{precision}f}')
				Y.append(mu)
				Y_errs.append(std)
				if verbose >= 50 and n_th==5:
					for idx_repeat, repeat in enumerate(repeat_vs):
						p = repeat['delta_X']
						n_th_centroids = repeat['history'][n_th-1]['centroids']
						print(f'{alg_name}, {data_name}, p={p}, {idx_repeat}th repeat, true:', fmt_np(repeat['true_centroids']), f' {n_th-1}_th:', fmt_np(n_th_centroids), ' final:', fmt_np(repeat['final_centroids']), )
				_tmp = [repeat['n_training_iterations'] for repeat in repeat_vs]
				_n_training_iterations = (
					float(f'{np.mean(_tmp):.{precision}f}'), float(f'{np.std(_tmp):.{precision}f}'))
				n_training_iterations.append(_n_training_iterations)
		# print(f'{alg_name}: X: {X}, Y:{Y}, Y_errs: {Y_errs}, n_training_iterations: {n_training_iterations}')
		vs = [f'{_mu:.5f}+/-{_std:.5f}' for _mu, _std in zip(Y, Y_errs)]
		print(f'{alg_name}: X: {X}, Y:{vs}, n_training_iterations: {n_training_iterations}')
		# plt.errorbar(X, Y, Y_errs)
		table_res.append(alg_name + ', ' + ','.join(vs))

		if alg_name == 'kmeans':
			if raw_n_th == np.inf: # final result
				label = f'K-Means'
			else:
				label = f'K-Means({init_method}): {np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
			color = 'blue'
			ecolor = 'tab:red'
		elif alg_name =='kmedian':
			if raw_n_th == np.inf:
				label = f'K-Median'
			else:
				label = f'K-Median({init_method}): {np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
			color = 'green'
			ecolor = 'tab:brown'
		elif alg_name =='kmedian_l1':
			if raw_n_th == np.inf:
				label = f'K-Median_L1'
			else:
				label = f'K-Median_L1({init_method}): {np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
			color = 'm'
			ecolor = 'tab:cyan'
		else:
			label = 'K-Median(Tukey)'
			color = 'm'
			ecolor = 'tab:cyan'
		plt.errorbar(X, Y, Y_errs, fmt=markers_fmt[i_alg], alpha=1.0,
		             capsize=3, color=color, ecolor=ecolor,
		             markersize=8, markerfacecolor='black',
		             label=label)
	print('\n'.join(table_res))
	font_size = 15
	plt.legend(loc='upper left', fontsize=font_size - 2)  # bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
	if error_method in ['centroid_diff', 'centroid_diff2']:
		ylabel = 'Diff' if error_method == 'centroid_diff' else 'Diff2'
		# ylabel = f'{ylabel}: ' + '$\\left||\mu-\mu^{*}\\right||$' + ': $ACD_{' + f'{n_th}' + '}$'  # start from index 1
		if raw_n_th==np.inf:
			ylabel = '$ACD^{*}$'  # converged results
		else:
			ylabel = '$ACD_{' + f'{n_th}' + 'th}$'  # start from index 1

		if case == 'diff2_outliers':
			xlabel = '$\sigma^2_{noise}$'
		else:
			xlabel = '$p(\%)$'
	else:
		ylabel = 'Average Misclustered Error' + ': $ACD_{' + f'{n_th}' + '}$'
		xlabel = '$p$'
	axes.set_ylabel(f'{ylabel}', fontsize=font_size)
	# axes.set_xlabel('$\Delta$')  # the distance between outlier and origin.
	axes.set_xlabel(xlabel, fontsize=font_size)  # x_noise's position.
	# X = [_v for _i, _v in enumerate(X) if _i != 1]
	axes.set_xticks(X)
	if case ==  'diff2_outliers':
		pass
	else:
		axes.set_xticklabels([int(v * 100) for v in X])
	axes.tick_params(axis='both', which='major', labelsize=font_size - 2)

	plt.tight_layout()
	with open(out_file, 'wb') as f:
		plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		plt.show()
	# plt.clf()
	plt.close(fig)


def plot_centroids(X, y_pred, initial_centroids, centroids, is_show=True, params={}, self=None):
	fig, axes = plt.subplots()  # (width, height)
	plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.3)
	plt.axvline(x=0)
	plt.axhline(y=0)
	title = params['OUT_DIR']
	if self:
		title += f', {self.n_training_iterations}th iteration'
	plt.title('\n'.join([title[i:i+60] for i in range(0, len(title), 60)]))
	for i in range(centroids.shape[0]):
		c = centroids[i]
		plt.scatter(c[0], c[1], marker='x', color='red')
		offset = 0.
		xytext = (c[0], c[1] + offset)
		axes.annotate(f'({c[0]:.2f}, {c[1]:.2f})', xy=(c[0], c[1]), xytext=xytext, fontsize=15, color='red',
		              ha='center', va='center',  # textcoords='offset points',
		              bbox=dict(facecolor='none', edgecolor='gray', pad=1),
		              # arrowprops=dict(arrowstyle="->", color='gray', shrinkA=1,
		              #                 connectionstyle="angle3, angleA=90,angleB=0")
		              )

	plt.tight_layout()
	out_file = os.path.join(params['OUT_DIR'], f'{self.n_training_iterations}th_iteration.png')
	with open(out_file, 'wb') as f:
		fig.savefig(f, dpi=600, bbox_inches='tight')
	if is_show: plt.show()
	# plt.clf()
	plt.close(fig)

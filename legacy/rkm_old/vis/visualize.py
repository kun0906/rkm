"""

"""
# Email: kun.bj@outllok.com
import os.path
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rkm.utils.common import fmt_np, timer

precision = 3

import copy, itertools
@timer
def align_centroids(centroids, true_centroids):
	print(f"{len(centroids)} centroids include {len(list(itertools.permutations(centroids)))} permutations")
	c1 = copy.deepcopy(true_centroids)
	# check which point is close to which true centroids.
	min_d = np.inf
	for c in list(itertools.permutations(centroids)):
		d = np.sum(np.sum(np.square(c - c1), axis=1), axis=0)
		if d < min_d:
			min_d = d
			best_centroids = np.asarray(copy.deepcopy(c))
	return best_centroids


@timer
def align_centroids_nearest(centroids, true_centroids):
	print(f"{len(centroids)} centroids include {len(list(itertools.permutations(centroids)))} permutations")
	# c1 = copy.deepcopy(true_centroids)
	# check which centroid is close to which true centroids.

	# max_i  min_j(estimated_i - true_j)
	best_centroids = np.zeros((centroids.shape))
	for i, c1 in enumerate(centroids):
		j = np.argmin(np.sum((c1 - true_centroids) ** 2, axis=1))
		best_centroids[j] = i

	return best_centroids

def plot_misclustered_errors(resutls, fig_kwargs, out_file='.png', title='', error_method = 'misclustered_error',
                             is_show=True, raw_n_th=5, verbose=10, case = None, init_method='random'):
	fig, axes, idx_axes = fig_kwargs['fig'], fig_kwargs['axes'], fig_kwargs['idx_axes']
	fig2, axes2 = plt.subplots(nrows=5, ncols=7, sharex=False,
	                         sharey=False, figsize=(15, 13))  # figsize=(8, 6) (width, height)
	# fig.suptitle(title  + ', centroids update')
	colors = ['blue', 'green', 'm', 'b', 'r', 'tab:brown', 'tab:green', 'orange', ]
	if raw_n_th==-1:
		raw_n_th = np.inf
	markers_fmt = ['*-', 'o-', '^-', 'v-', 'v+']
	table_res = []
	py_names = [
		'kmeans',
		'kmedian_l1',
		'kmedian',  # our method
		# 'my_spectralclustering',
		# 'kmedian_tukey',
	]
	df = pd.DataFrame()
	for i_alg, alg_name in enumerate(py_names): # enumerate(resutls.keys()): algorithm
		X = []
		Y = []
		Y_errs = []
		for data_name, diff_outliers in resutls[alg_name].items(): # different datasets
			n_training_iterations = []
			_df = pd.DataFrame()
			for i_detail, (detail_name, delta_X_vs) in enumerate(diff_outliers.items()): # for each delta_X or param
				try:
					p = delta_X_vs['SEED_0']['SEED2_0']['delta_X']  # delta_X is alias of p
				except Exception as e:
					p = delta_X_vs['Seed_0']['Seed2_0']['delta_X']
				X.append(p)

				n_ths = []
				acd_metrics = []
				seed1s = []
				true_centroids = []
				initial_centroids = []
				final_centroids = []
				for i_seed1, (seed1, seed1_vs) in enumerate(delta_X_vs.items()): # for seed1 (outloop seed)
					seed1s.append(seed1)
					_n_ths = []
					_acds = []
					for i_seed2, (seed2, seed2_vs) in enumerate(seed1_vs.items()): # for seed2 (innerloop seed)
						# if raw_n_th > the total iterations, then use the final results.
						_n_th = len(seed2_vs['history'])
						n_th = min(raw_n_th, _n_th)
						_n_ths.append(n_th)

						_centroids = seed2_vs['history'][n_th - 1]['centroids']
						_true_centroids = seed2_vs['data']['true_centroids']

						if init_method != 'omniscient':
							# takes much time due to the permutation
							_centroids = align_centroids(_centroids,
														 _true_centroids)  # align the centroids with ground truth
						else:
							print('+++no alignment for omniscient case.')
						if error_method == 'max_centroid_diff':
							_acd = max(np.sum(np.square(_centroids - _true_centroids), axis=1))
						else:
							if error_method == 'average_centroid_diff':
								# not align
								# _acd = seed2_vs['history'][n_th - 1]['scores']['centroid_diff']
								_acd = np.mean(np.sum(np.square(_centroids - _true_centroids), axis=1))
							else:
								raise NotImplementedError(error_method)

						_true_centroids = seed2_vs['data']['true_centroids']
						_initial = seed2_vs['history'][0]['centroids']
						_final = seed2_vs['history'][n_th - 1]['centroids']
						true_centroids.append(_true_centroids.tolist())
						initial_centroids.append(_initial.tolist())
						final_centroids.append(_final.tolist())
						_acds.append(_acd)
						if verbose>=10:
							if _acd >= 0.0:
								print(f'{alg_name},{init_method}, {data_name},{detail_name},p:{p},{seed1},{seed2}, final:{fmt_np(_final)}, initial:{fmt_np(_initial)},ACD:{_acd:.3f}')

					# plot each seed's results (100 repeats per seed)
					# only plot the first seed1
					_mu, _std = np.mean(_acds), np.std(_acds)
					_acds_fmt = [ f'{_v:.2f}' for _v in _acds]
					print(f'***{alg_name},{init_method},p:{p},seed1:{seed1},# of seeds2:{i_seed2 + 1}, _mu:{_mu:.2f}, _std:{_std:.2f}, _acds: {_acds_fmt}, _n_ths: {_n_ths}')
					if verbose>=50 and seed1 == 'SEED_0' and i_detail < 7:  # seed1 == 'SEED_11000'
						# plot all in one
						# p = delta_vs[0]['delta_X']
						# axes2[0, i_repeat].hist(acd_metrics, color=colors[i_alg], label=f'{alg_name}: {data_name}: {p}')
						axes2[0, i_detail].hist(_acds, color=colors[i_alg], label=f'{alg_name}:{_mu:.2f}+/-{_std:.2f}', alpha=0.5, density=False)
						# if i_delta == 0:
						axes2[0, i_detail].legend(loc='upper right', fontsize=6)
						axes2[0, i_detail].set_title(f'X_noise:{p}\nseed1:{seed1}\n# of seeds2:{i_seed2+1}')
						axes2[0, i_detail].set_xlabel('ACD')
						axes2[0, i_detail].set_ylabel('Frequency')

						# plot each algorithm's result
						# p = delta_vs[0]['delta_X']
						# axes2[i_repeat].hist(acd_metrics, color=colors[i_alg], label=f'{alg_name}: {data_name}: {p}')
						axes2[i_alg+1, i_detail].hist(_acds, color=colors[i_alg], label=f'{alg_name}:{_mu:.2f}+/-{_std:.2f}', alpha=0.5, density=False)
						# if i_delta == 0:
						axes2[i_alg+1, i_detail].legend(loc='upper right', fontsize=6)
						axes2[i_alg+1,i_detail].set_title(f'X_noise:{p}')
						axes2[i_alg+1, i_detail].set_xlabel(f'ACD') #:{mu}+/-{std}
						axes2[i_alg+1, i_detail].set_ylabel('Frequency')

					n_ths.append([np.mean(_n_ths), np.std(_n_ths), _n_ths])
					acd_metrics.append([np.mean(_acds), np.std(_acds), _acds])

				# print(alg_name, data_name, detail_name, seed1s, acd_metrics, n_ths)
				if verbose >= 2: print(alg_name, data_name, detail_name)
				# df = pd.DataFrame(zip(seed1s, acd_metrics, n_ths), columns=['seed1s', 'ACD(mu+/-std)', 'n_ths(mu+/-std)'])
				_df2 = pd.DataFrame({'seed1s': seed1s,
				                   'ACD(mu+/-std, ACDs)': acd_metrics,
				                   'n_th(mu+/-std, n_ths)':n_ths,
									 'true': true_centroids,
									 'initial':initial_centroids,
									'final': final_centroids}
				                 )
				if verbose >= 2: print(_df2)
				_df['alg_name'] = alg_name
				_df['data_name'] = data_name
				_df['init_method'] = init_method
				_df[[f'{p}:{_c}' for _c in _df2.columns]] = _df2
				# plot the mu+/-std results
				_tmp = [_mu for _mu, _std, _ in acd_metrics]
				mu = float(f'{np.mean(_tmp):.{precision}f}')
				# std = float(f'{np.std(_tmp):.{precision}f}')
				std = float(f'{np.std(_tmp):.{precision}f}') / np.sqrt(len(_tmp))	# standard error
				Y.append(mu)
				Y_errs.append(std)

				_tmp = [_mu for _mu, _std, _ in n_ths]
				_n_training_iterations = (
					float(f'{np.mean(_tmp):.{precision}f}'), float(f'{np.std(_tmp):.{precision}f}'))
				n_training_iterations.append(_n_training_iterations)

			if df.size==0:
				df = _df.copy(deep=True)
			else:
				empty = pd.DataFrame([[None] * df.shape[1]])
				df = pd.concat([df, empty, _df], axis=0)
			# df.to_csv(out_file + '.csv')
			try:
				df.to_csv(out_file + '.csv')
			except Exception as e:
				print(e)
				traceback.print_exc()
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
		elif alg_name =='my_spectralclustering':
			if raw_n_th == np.inf:
				label = f'SpectralClustering'
			else:
				# label = f'K-Median_L1({init_method}): {np.mean(n_ths):.2f}+/-{np.std(n_ths):.2f}'
				label = f'SpectralClustering(Discretize,p_{p}):{mu:.2f}+/-{std:.2f}'
			color = 'brown'
			ecolor = 'tab:cyan'
		else:
			label = f'{alg_name}'
			color = 'm'
			ecolor = 'tab:cyan'
		if idx_axes < len(fig.axes)-1: label = ''		# for the first two plots, we don't have legends
		h = axes.errorbar(X, Y, Y_errs, fmt=markers_fmt[i_alg],
		             capsize=3, color=color, ecolor=ecolor,
		             markersize=8, markerfacecolor='black',
		             label=label, alpha=1)

	print('\n'.join(table_res))
	font_size = 15
	# # axes.legend(loc='upper left', fontsize=font_size - 2)  # bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
	# # Shrink current axis's height by 10% on the bottom
	# box = axes.get_position()
	# axes.set_position([box.x0, box.y0 + box.height * 0.01,
	# 				 box.width, box.height * 0.99])
	# # Put a legend below current axis
	# axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False,
	# 		  fancybox=False, shadow=False, ncol=3, fontsize=font_size-2)

	# if error_method in ['centroid_diff', 'centroid_diff2']:
	# 	ylabel = '$\\left||\mu-\mu^{*}\\right||$' + f'$ACD_{n_th-1}$'
	# else:
	# 	ylabel = 'Average Misclassified Error (AME)'
	if error_method in ['average_centroid_diff', 'centroid_diff2']:
		# ylabel = 'Diff' if error_method == 'centroid_diff' else 'Diff2'
		# ylabel = f'{ylabel}: ' + '$\\left||\mu-\mu^{*}\\right||$' + ': $ACD_{' + f'{n_th-1}' + '}$'
		# ylabel = '$ACD_{' + f'{n_th}' + 'th}$'   # start from index 1
		if raw_n_th==np.inf:
			ylabel = '$ACD:\mu\pm\\frac{\sigma}{\sqrt{m}}$'  # converged results
		else:
			ylabel = '$ACD_{' + f'{n_th}' + 'th}$'  # start from index 1

		# xlabel = '$x_{noise}$'
	elif error_method in ['max_centroid_diff']:
		# ylabel = 'Diff' if error_method == 'centroid_diff' else 'Diff2'
		# ylabel = f'{ylabel}: ' + '$\\left||\mu-\mu^{*}\\right||$' + ': $ACD_{' + f'{n_th-1}' + '}$'
		# ylabel = '$ACD_{' + f'{n_th}' + 'th}$'   # start from index 1
		if raw_n_th==np.inf:
			ylabel = '$MCD:\mu\pm\\frac{\sigma}{\sqrt{m}}$'  # converged results
		else:
			ylabel = '$MCD_{' + f'{n_th}' + 'th}$'  # start from index 1

		# xlabel = '$x_{noise}$'
	else:
		ylabel = 'Average Misclustered Error' + ': $ACD_{' + f'{n_th}' + '}$'   # start from index 1
		# xlabel = '$x_{noise}$'
	xlabel = fig_kwargs['xlabel']
	if idx_axes == 0: axes.set_ylabel(f'{ylabel}', fontsize=font_size)
	axes.set_xlabel(xlabel, fontsize=font_size)  # the distance between outlier and origin.
	# X = [_v for _i, _v in enumerate(X) if _i != 1]
	axes.set_xticks(X)
	axes.tick_params(axis='both', which='major', labelsize=font_size-2)
	# axes.xticks(fontsize=font_size-2)
	# if alg_name == 'my_spectralclustering':
	# 	axes.set_title('None', fontsize=font_size - 2)
	# else:
	idx2title = {0:'(a) Random', 1: '(b) K-Means++', 2:'(c) Omniscient'}
	axes.set_title(idx2title[idx_axes], fontsize=font_size-2)

	if idx_axes == len(fig.axes)-1:
		# # Shrink current axis's height by 10% on the bottom
		# box = fig.get_position()
		# fig.set_position([box.x0, box.y0 + box.height * 0.01,
		# 				 box.width, box.height * 0.99])
		# # # Put a legend below current axis
		# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False,
		# 		  fancybox=False, shadow=False, ncol=3, fontsize=font_size-2)

		fig.legend(handles=axes.get_legend(), loc="upper center", bbox_to_anchor=[0.5, 1.01],
				   ncol=4, shadow=False, title="", fancybox=False, frameon=False)
		fig.tight_layout(rect=[0, 0, 1, 0.95])		# (left, bottom, right, top)
		with open(out_file+'.png', 'wb') as f:
			fig.savefig(f, dpi=600, bbox_inches='tight')
		if is_show:
			fig.show()
		# plt.clf()
		plt.close(fig)

	# try:
	# 	df.to_csv(out_file + '.csv')
	# except Exception as e:
	# 	print(e)
	# 	traceback.print_exc()
	if raw_n_th == -1 or raw_n_th == np.inf:
		title = '$ACD_{' + '*' + '}$'
	else:
		title = '$ACD_{' + f'{raw_n_th}' + '}$'
	# fig2.suptitle(title)
	# fig2.tight_layout()
	# with open(out_file[:-4]+'_hist.png', 'wb') as f:
	# 	fig2.savefig(f, dpi=600, bbox_inches='tight')
	# if is_show:
	# 	fig2.show()
	plt.close(fig2)

	return df.values



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
	is_show = False
	if is_show: plt.show()
	# plt.clf()
	plt.close(fig)

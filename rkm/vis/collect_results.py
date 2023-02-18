"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/rkm/rkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 vis/collect_results.py
"""
# Email: kun.bj@outllok.com
import collections
import copy
import os
import traceback

from rkm import config
from rkm.main_all import get_datasets_config_lst, get_algorithms_config_lst
from rkm.utils.common import load, check_path
from rkm.vis.visualize import plot_misclustered_errors, plot_mixed_clusters

n_precision = 3


def _parser_history(args):
	OUT_DIR = args['OUT_DIR']
	results = []
	try:
		out_dat = os.path.join(OUT_DIR, 'history.dat')
		history = load(out_dat)
		results = {'delta_X': history['delta_X'], 'misclustered_error': history['scores']['misclustered_error'],
		           'centroid_diff': history['scores']['centroid_diff'],
		           'true_centroids': history['data']['true_centroids'],
		           'init_centroids': history['data']['init_centroids'],
		           'final_centroids': history['history'][-1]['centroids'],
		           'n_training_iterations': len(history['history']),
		           'history': history['history']}
		# print(len(history['history']), history['delta_X'], history['scores']['misclustered_error'])
	except Exception as e:
		traceback.print_exc()

	return results


def parser_history(args_dict):
	results_detail = {}
	for i_args, (seed2, args) in enumerate(args_dict.items()):
		try:
			results = _parser_history(args)
			results_detail[seed2] = results
		except Exception as e:
			traceback.print_exc()

	return results_detail


def save2xls(workbook, worksheet, column_idx, args, results_avg, metric_names):
	dataset_name = args['DATASET']['name']
	dataset_detail = args['DATASET']['detail']
	# algorithm_name = args['ALGORITHM']['name']
	algorithm_py_name = args['ALGORITHM']['py_name']
	algorithm_detail = args['ALGORITHM']['detail']
	# print(params)
	OUT_DIR = args['OUT_DIR']
	# print(f'OUT_DIR: {OUT_DIR}')
	# set background color
	cell_format = workbook.add_format()
	cell_format.set_pattern(1)  # This is optional when using a solid fill.
	cell_format.set_text_wrap()
	# new cell_format to add more formats to one cell
	cell_format2 = copy.deepcopy(cell_format)
	cell_format2.set_bg_color('FF0000')
	# cell_format.set_bg_color('#FFFFFE')

	row = 0
	# add dataset details, e.g., plot
	worksheet.set_row(row, 100)  # set row height to 100
	if column_idx == 1:
		scale = 0.248
		dataset_img = os.path.join(OUT_DIR, dataset_detail + '.png')
		if os.path.exists(dataset_img):
			worksheet.insert_image(row, column_idx, dataset_img,
			                       {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	row += 1

	# Widen the first column to make the text clearer.
	worksheet.set_row(row, 100)  # set row height to 100
	worksheet.set_column(column_idx, column_idx, width=50, cell_format=cell_format)
	# Insert an image offset in the cell.
	if column_idx == 0:
		s = f'{OUT_DIR}'
	else:
		s = ''
	worksheet.write(row, column_idx, s)
	if column_idx == 1:
		dataset_img = os.path.join(OUT_DIR, dataset_detail + '-' + args['NORMALIZE_METHOD'] + '.png')
		if os.path.exists(dataset_img):
			worksheet.insert_image(row, column_idx, dataset_img,
			                       {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	row += 1

	# write the second row
	s = f'{algorithm_py_name}'
	steps = 2
	start = column_idx // steps
	colors = ['#F0FEFE', '#FEF0FE', '#FEFEF0', '#F0FBFB', '#FBF0FB', '#FBFBF0']
	if column_idx == 0:
		cell_format.set_bg_color('#F0FEFE')
	elif column_idx == 1:
		cell_format.set_bg_color('#FEF0FE')
	elif start * steps <= column_idx < start * steps + steps:
		cell_format.set_bg_color(colors[start % len(colors) + 2])  # #RRGGBB
	else:
		cell_format.set_bg_color('#FFFFFF')  # white
	worksheet.write(row, column_idx, s)
	row += 1
	# Insert an image offset in the cell.
	s = f'{dataset_name}\n{dataset_detail}\n{algorithm_py_name}\n{algorithm_detail}\n'
	worksheet.set_row(row, 60)  # set row height to 100
	worksheet.write(row, column_idx, s)
	row += 1
	try:
		for split in args['SPLITS']:
			# metric_names = results_avg[split]['metric_names']
			s = f'*{split}:\n'
			# if split == 'train':
			# 	Iterations = results_avg[split]['Iterations']
			# 	durations = results_avg[split]['durations']
			# 	s += f'\titerations: {Iterations[0]:.2f} +/- '\
			# 	     f'{Iterations[1]:.2f}\n'
			# 	s += f'\tdurations: {durations[0]:.2f} +/- ' \
			# 	     f'{durations[1]:.2f}\n'
			# else:
			# 	s += '\n'
			for metric_name in metric_names:

				if metric_name in ['n_clusters', 'n_clusters_pred']:
					value = results_avg[split][metric_name]
					s += f'\t{metric_name}({len(value)}):\n{value}\n'

				elif metric_name in ['labels_pred', 'labels_true']:
					value = results_avg[split][metric_name]
					s += f'\t{metric_name}({len(value)}):\n'
					s += '\n'.join([str(v) for v in value]) + '\n'
					tmp_ = sum(value, [])
					tmp_ = sum([[f'{k_}:{v_}' for k_, v_ in d_.items()] for d_ in tmp_], [])  # flatten a nested list
					tmp_ = collections.Counter(tmp_)
					tmp_ = sorted(tmp_.items(), key=lambda kv: kv[1], reverse=True)
					# tmp_ = '\n'.join([f'{k_}, {v_}' for k_, v_ in tmp_])
					s += f'\tdistribution:\n{tmp_}\n'

				else:
					score_mean, score_std = results_avg[split][metric_name]
					s += f'\t{metric_name}: {score_mean:.2f} +/- {score_std:.2f}\n'

			worksheet.set_row(row, 400)  # set row height to 100
			worksheet.write(row, column_idx, s)
			row += 1

			s = ''
			init_centroids = results_avg[split]['init_centroids']
			final_centroids = results_avg[split]['final_centroids']
			final_centroids_lst = results_avg[split]['final_centroids_lst']
			s += f'init_centroids:\n'
			s += '\n'.join([str(v) for v in init_centroids]) + '\n'
			s += f'final_centroids:\n'
			s += '\n'.join([str(v) for v in final_centroids]) + '\n'
			# final centroids distribution
			s += 'final centroids distribution: \n'
			ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
			tot_centroids = len(final_centroids_lst)
			s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
			                 cen_, cnt_ in ss_)

			data = s
			# Insert an image offset in the cell.
			worksheet.set_row(row, 400)  # set row height to 100
			# cell_format = workbook.add_format({'bold': True, 'italic': True})
			# cell_format2 = workbook.add_format()
			cell_format.set_align('top')
			worksheet.write(row, column_idx, data, cell_format)
			row += 1
	# break
	except Exception as e:
		print(f'Error: {e}')
		traceback.print_exc()
		data = '-'

	# # worksheet.write('A12', 'Insert an image with an offset:')
	# n_clients = 0 if 'Centralized' in algorithm_py_name else args['N_CLIENTS']
	# sub_dir = f'Clients_{n_clients}'
	# centroids_img = os.path.join(out_dir, sub_dir, f'M={n_clients}-Centroids.png')
	# print(f'{centroids_img} exist: {os.path.exists(centroids_img)}')
	# worksheet.set_row(row, 300)  # set row height to 30
	scale = 0.248
	# if os.path.exists(centroids_img):
	# 	worksheet.insert_image(row, column_idx, centroids_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	# row += 1

	score_img = os.path.join(OUT_DIR, 'over_time', f'centroids_diff.png')
	print(f'{score_img} exist: {os.path.exists(score_img)}')
	worksheet.set_row(row, 300)  # set row height to 30
	if os.path.exists(score_img):
		worksheet.insert_image(row, column_idx, score_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	row += 1


def main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=False, IS_GEN_DATA=True, VERBOSE=5, CASE='', init_method='random'):
	# get default config.yaml
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = args['OUT_DIR']
	SEPERTOR = args['SEPERTOR']
	args['N_REPEATS'] = N_REPEATS
	args['OVERWRITE'] = OVERWRITE
	args['VERBOSE'] = VERBOSE

	tot_cnt = 0
	dataset_names = ['3GAUSSIANS']  #  '3GAUSSIANS'
	py_names = [
		# 'kmeans',
		'kmedian_l1',
		# 'kmedian',
		# 'kmedian_tukey',
	]

	results = {}
	datasets = get_datasets_config_lst(dataset_names, case=CASE)
	for dataset in datasets:
		algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		for i_alg, algorithm in enumerate(algorithms):
			if algorithm['init_method'] != init_method: continue
			if VERBOSE > 10: print(f'\n*** {tot_cnt}th experiment ***:', dataset['name'], algorithm['py_name'])
			for i_repeat in range(N_REPEATS):
				seed_step = 1000
				seed = i_repeat * seed_step  # data seed
				seed_step2 = seed_step // 100  # repeats 10 times in the inner loop
				args_dict = {}
				for seed2 in range(seed, seed + seed_step, seed_step2):
					if VERBOSE >=10: print('***', dataset['name'], i_repeat, seed, seed2)
					args1 = copy.deepcopy(args)
					args1['SEED_1'] = seed2
					args1['SEED_DATA'] = seed2
					args1['DATASET']['name'] = dataset['name']
					args1['DATASET']['detail'] = dataset['detail']
					args1['N_CLUSTERS'] = dataset['n_clusters']
					N_REPEATS = args1['N_REPEATS']
					N_CLUSTERS = args1['N_CLUSTERS']
					NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
					args1['DATASET']['detail'] = os.path.join(f'{SEPERTOR}'.join([args1['DATASET']['detail'],
					                                                              NORMALIZE_METHOD, f'K_{N_CLUSTERS}']),
					                                           f'SEED_{seed}', f'SEED2_{seed2}')
					dataset_name = args1['DATASET']['name']
					dataset_detail = args1['DATASET']['detail']
					args1['data_file'] = os.path.join(args1['IN_DIR'], dataset_name, f'{dataset_detail}.dat')
					if VERBOSE >= 10: print(f'arg1.data_file:', args1['data_file'])
					args1['ALGORITHM']['py_name'] = algorithm['py_name']
					args1['ALGORITHM']['init_method'] = algorithm['init_method']
					init_method = args1['ALGORITHM']['init_method']
					NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
					args1['ALGORITHM']['detail'] = f'{SEPERTOR}'.join([f'R_{N_REPEATS}',
					                                                   f'{init_method}',
					                                                   f'{NORMALIZE_METHOD}'])
					args1['OUT_DIR'] = os.path.join(OUT_DIR, args1['DATASET']['name'], f'{dataset_detail}',
					                                args1['ALGORITHM']['py_name'], args1['ALGORITHM']['detail'])
					args_dict[f'Seed2_{seed2}'] = copy.deepcopy(args1)

				try:
					results_detail = parser_history(args_dict)
				except Exception as e:
					traceback.print_exc()
				tot_cnt += 1

				alg_key = algorithm['py_name']
				data_key = dataset['name']
				detail = dataset['detail']
				if alg_key not in results.keys():
					results[alg_key] = {}
				if data_key not in results[alg_key].keys():
					results[alg_key][data_key] = {}
				if detail not in results[alg_key][data_key].keys():
					results[alg_key][data_key][detail] = {f'Seed_{seed}': results_detail}

				results[alg_key][data_key][detail][f'Seed_{seed}'] = results_detail

	# print(results.items())
	print(f'*** Total cases: {tot_cnt}')

	for n_th in [-1]: #[1, 2, 3, 4, 5, 10, -1]:
		# show  error at the n_th iteration
		# 'centroid_diff2' is the next round iteration of centroid_diff,
		# e.g., the ith differences of centroid_diff2 is the (i+1)th differences of centroid_diff.
		for error_method in ['centroid_diff']: #  'misclustered_error'
			# if CASE in ['diff_outliers', 'diff2_outliers']:
			# 	out_file = os.path.join(OUT_DIR, 'xlsx', args1['DATASET']['name'],
			# 	                        f'{os.path.dirname(dataset_detail)}',
			# 	                        args1['ALGORITHM']['detail'] + f'-diff_outliers-{error_method}-{n_th}th.png')
			# 	check_path(out_file)
			# 	plot_misclustered_errors(results, out_file, error_method=error_method, is_show=True, raw_n_th=n_th,
			# 	                         verbose=10, case = CASE, init_method=init_method)
			# 	print(out_file)
			# else:
			# 	out_file = os.path.join(OUT_DIR, 'xlsx', args1['DATASET']['name'],
			# 	                        f'{os.path.dirname(dataset_detail)}',
			# 	                        args1['ALGORITHM']['detail'] + f'-mixed_clusters-{error_method}-{n_th}th.png')
			# 	check_path(out_file)
			# 	plot_mixed_clusters(results, out_file, error_method=error_method, is_show=True, raw_n_th=n_th,
			# 	                    verbose=10, case = CASE, init_method=init_method)  # show misclustered error at the n_th iteration
			# 	print(out_file)
			if error_method == 'centroid_diff':
				metric = 'ACD'
			else:
				raise NotImplementedError(error_method)
			if CASE == 'diff_outliers':
				f = f'Case1_' + args1['ALGORITHM']['detail'] + f'_{metric}_{n_th}th.png'
			elif CASE=='diff2_outliers':
				f = f'Case2_' + args1['ALGORITHM']['detail'] + f'_{metric}_{n_th}th.png'
			else:
				raise NotImplementedError(CASE)
			f = f.replace('|','_')
			out_file = os.path.join(OUT_DIR, 'xlsx', args1['DATASET']['name'],
			                        f'{os.path.dirname(os.path.dirname(dataset_detail))}', f)
			check_path(out_file)
			plot_misclustered_errors(results, out_file, error_method=error_method, is_show=True, raw_n_th=n_th,
			                         verbose=VERBOSE, case=CASE, init_method=init_method)
			print(out_file)

if __name__ == '__main__':
	cases = ['diff_outliers', 'diff2_outliers', 'constructed2_3gaussians', 'constructed_3gaussians']
	cases = ['diff2_outliers']
	for CASE in cases: # , 'constructed_3gaussians', 'diff2_outliers', 'diff_outliers', 'mixed_clusters']: # ['diff_outliers', 'mixed_clusters', 'constructed_3gaussians']:  # , 'mixed_clusters'
		for init_method in ['random', 'kmeans++', 'omniscient',]: #['random', 'kmeans++', 'omniscient',]:
			try:
				main(N_REPEATS=20, OVERWRITE=True, IS_DEBUG=True, VERBOSE=2, CASE=CASE, init_method=init_method)
			except Exception as e:
				traceback.print_exc()

""" Run this main file for a single experiment

	Run instruction:
	$pwd
	$rkm/rkm
	$PYTHONPATH='..' python3 main_single.py
"""
# Email: kun.bj@outllok.com

from pprint import pprint

from rkm import config
from rkm import datasets
from rkm.cluster.kmeans import KMeans
from rkm.cluster.kmedian import KMedian


class Framework:

	def __init__(self, args):
		self.args = args

	def run(self):

		self.data = datasets.gen_data(self.args)
		X, y = self.data

		ALG2PY = {'kmeans': KMeans,
		          'kmedian': KMedian,
		          }

		self.model = ALG2PY[self.args['ALGORITHM']['name']]

		self.model.fit(X, y)

		self.history = self.model.history

		# save results

	def vis(self):
		pass


def main(config_file='config.yaml'):
	"""

	Parameters
	----------
	config_file

	Returns
	-------

	"""

	# args = config.parser(config_file)
	#
	# X, y = datasets.gen_data(args)
	#
	#
	# ALG2PY = {'kmeans': KMeans,
	#           'kmedian': KMedian,
	#           }
	# model = ALG2PY[args['ALGORITHM']['name']]
	#
	# model.fit(X, y)
	#
	# history  = model.history
	#
	# return history

	args = config.parser(config_file)
	fw = Framework(args)
	fw.run()

	return fw


	# # Step 0: config the experiment
	# args = config.parser(config_file)
	# if args['VERBOSE'] >= 2:
	# 	print(f'~~~ The template config {config_file}, which will be modified during the later experiment ~~~')
	# 	pprint(args, sort_dicts=False)
	#
	# # Step 1: run cluster and get result
	# history_file = _main.run_model(args)
	# args['history_file'] = history_file
	#
	# # Step 2: visualize the result
	# visual_file = visualize.visualize_data(args)
	# args['visual_file'] = visual_file
	#
	# # # Step 3: dump the config
	# # config.dump(args['config_file'][:-4] + 'out.yaml', args)
	#
	# return args



if __name__ == '__main__':
	args = main(config_file='config.yaml')
	pprint(args, sort_dicts=False)

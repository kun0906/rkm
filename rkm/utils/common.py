"""
	Common used functions
"""
# Email: kun.bj@outllok.com

import os
import pickle
import time
from datetime import datetime


def load(in_file):
	with open(in_file, 'rb') as f:
		data = pickle.load(f)
	return data


def dump(data, out_file):
	out_dir = os.path.dirname(out_file)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	with open(out_file, 'wb') as out:
		pickle.dump(data, out)


def timer(func):
	# This function shows the execution time of
	# the function object passed
	def wrap_func(*args, **kwargs):
		t1 = time.time()
		print(f'{func.__name__}() starts at {datetime.now()}')
		result = func(*args, **kwargs)
		t2 = time.time()
		print(f'{func.__name__}() ends at {datetime.now()}')
		print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
		return result

	return wrap_func


def check_path(file):
	# # file = fr"{file}"
	# file = os.path.expanduser(file).encode('unicode_escape')
	# if os.path.isfile(file):
	# 	tmp_dir = os.path.dirname(file)
	if os.path.isdir(file):
		tmp_dir = file
	else:
		tmp_dir = os.path.dirname(file)

	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)

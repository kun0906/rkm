import traceback
import time
import os

def gen_all_sh(args_lst):
	"""

	Parameters
	----------
	py_name
	case

	Returns
	-------

	"""
	args = args_lst[0]
	# check_arguments()
	dataset_name = args['DATASET']['name']
	dataset_detail = os.path.dirname(args['DATASET']['detail'])
	# n_clients = args['N_CLIENTS']
	# algorithm_py_name = args['ALGORITHM']['py_name']
	# # algorithm_name = args['ALGORITHM']['name']
	# algorithm_detail = args['ALGORITHM']['detail']
	# # n_clusters = args['ALGORITHM']['n_clusters']
	# config_file = args['config_file']
	# OUT_DIR = os.path.join(*args['OUT_DIR'].split('/')[:4])
	OUT_DIR = args['OUT_DIR']

	job_name = f'{dataset_name}-{dataset_detail}'
	# tmp_dir = '~tmp'
	# if not os.path.exists(tmp_dir):
	# 	os.system(f'mkdir {tmp_dir}')
	if '2GAUSSIANS' in dataset_name:
		t = 48
	# elif 'FEMNIST' in dataset_name and 'greedy' in algorithm_py_name:
	# 	t = 48
	else:
		t = 48
	content = fr"""#!/bin/bash
#SBATCH --job-name={OUT_DIR}         # create a short name for your job
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={OUT_DIR}/out.txt
#SBATCH --error={OUT_DIR}/err.txt
### SBATCH --mail-type=end          # send email when job ends
###SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification. \

## SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
## SBATCH --output={OUT_DIR}/%j-{job_name}-out.txt
## SBATCH --error={OUT_DIR}/%j-{job_name}-err.txt
## SBATCH --nodes=1                # node count
## SBATCH --ntasks=1               # total number of tasks across all nodes
## SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
## SBATCH --mem=40G\
### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-user=kun.bj@cloud.com # not work 
### SBATCH --mail-user=<YourNetID>@princeton.edu\

module purge
module load anaconda3/2021.11   # Python 3.9.7
pip3 install nltk 

cd /scratch/gpfs/ky8517/rkm/rkm 
pwd
python3 -V

"""
	content += "\nexport PYTHONPATH=\"${PYTHONPATH}:..\" \n"
	for i, args in enumerate(args_lst):
		config_file_ = args['config_file']
		out_dir_ = args['OUT_DIR']
		# content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {algorithm_py_name} --dataset '{dataset_name}' " \
		#                   f"--data_details '{dataset_detail}' --algorithm '{algorithm_name}' --n_clusters '{n_clusters}' --n_clients '{n_clients}' \n"
		content += f"\nPYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py > '{out_dir_}/a.txt' 2>&1  &\n"
		# if i == 0: 	# let the first one run first to generate the data and avoid unknown conflict for the rest.
		# 	content += "echo 'Finish the first one first...'\n"
		content += "\nwait\n"  # must has this line
		content += "echo $!\n"  # stores the background process PID
		content += "echo $?\n"  # $? stores the exit status.
		content += f"echo \'Finish {i}-th args.\'"  # $? stores the exit status.
	# single quote (just treat the content inside it as a string) vs double quote in bash
	# $ echo "$(echo "upg")"
	# upg
	# $ echo '$(echo "upg")'
	# $(echo "upg")
	#
	content += "\nwait\n"  # must has this line
	# The bash wait command is a Shell command that waits for background running processes to complete and returns the exit status.
	# Without any parameters, the wait command waits for all background processes to finish before continuing the script.
	content += "echo $!\n"  # stores the background process PID
	content += "echo $?\n"  # $? stores the exit status.
	content += "\necho \'done\n"
	# sh_file = f'{OUT_DIR}/{dataset_name}-{dataset_detail}-{algorithm_name}-{algorithm_detail}.sh'
	sh_file = f'{OUT_DIR}/sbatch.sh'
	with open(sh_file, 'w') as f:
		f.write(content)
	cmd = f"sbatch '{sh_file}'"
	print(cmd)
	os.system(cmd)


def main():
	st = time.time()
	# No need to create dataset, OVERWRITE=False
	# cases = ['diff_outliers', 'diff2_outliers', 'constructed_3gaussians', 'constructed2_3gaussians']
	cases = ['diff2_outliers']
	py_names = [
		'kmeans',
		'kmedian_l1',
		'kmedian',  # our method
		# 'kmedian_tukey',
	]
	for CASE in cases:  # , 'mixed_clusters', 'diff_outliers', 'constructed_3gaussians', 'constructed2_3gaussians
		for py_name in py_names:
			try:
				main(N_REPEATS=2, OVERWRITE=False, IS_DEBUG=True, VERBOSE=1, CASE=CASE, py_names=[py_name])
			except Exception as e:
				traceback.print_exc()
	ed = time.time()
	print(f'Total: {ed-st}s.')

if __name__ == '__main__':
	main()

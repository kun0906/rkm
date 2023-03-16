"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/rkm/rkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 vis/latex_tables.py > table.txt
"""
# Email: kun.bj@outllok.com
import os

import numpy as np
import pandas as pd

ALG_NAMES = {
    'kmeans': 'K-Means',
    'kmedian_l1': 'K-Median-L1',
    'kmedian': 'K-Median(our)',  # our method
    # # 'kmedian_tukey',
    'my_spectralclustering': 'Spectral',
}
INIT_NAMES = {'random': 'Random',
              'kmeans++': 'K-Means++',
              'omniscient': 'Omniscient'
              }


def _main(in_files, ps, py_names, precision=3, verbose=0):
    # in_file = '/Users/kun/Downloads/rkm/20230312/Case4_R_1000_None_NONE_ACD_-1th.png.csv'
    # ps = [0.05, 0.1, 0.2, 0.35, 0.49]
    for metric in ['ACD', 'n_th']:
        print(f'---Show {metric}')
        for i_file, in_file in enumerate(in_files):
            df = pd.read_csv(in_file)
            if 'random' in in_file:
                init_method = 'random'
            elif 'kmeans++' in in_file:
                init_method = 'kmeans++'
            elif 'omniscient' in in_file:
                init_method = 'omniscient'
            else:
                raise NotImplemented(in_file)
            if verbose >= 2: print(f'\n***{metric}, {in_file}, {ps}')
            for i_alg, alg_name in enumerate(py_names):
                mask = (df['alg_name'] == alg_name)
                alg_df = df[mask]
                line = []
                p_line = []
                for p in ps:
                    # col = f'{p}:ACD(mu+/-std, ACDs)'
                    # col = f'{p}:n_th(mu+/-std, n_ths)'
                    col = f'{p}:{metric}(mu+/-std, {metric}s)'  # number of iteration for each experimental setting.
                    _df = alg_df.loc[:, col].apply(lambda lst: float(lst.split(',')[0][1:]))
                    n_repeats = len(_df)
                    if verbose >= 5:
                        print(_df)
                    _mu = np.mean(_df)
                    _std = np.std(_df)
                    line.append(f'{_mu:.{precision}f} $\pm$ {_std/np.sqrt(n_repeats):.{precision}f}')
                    p_line.append(str(p))
                line = '  &  '.join(line) + r' \\'
                p_line = '  &  '.join(p_line) + r' \\'
                if i_file == 0 and i_alg == 0:
                    print(r'\toprule')
                    print('     &      & ' + p_line)
                    print('\midrule')

                if i_alg == 0:
                    line = ALG_NAMES[alg_name] + '  &  ' + '\multirow{3}{*}{'  + INIT_NAMES[init_method]+ '}  & ' + line
                else:
                    line = ALG_NAMES[alg_name] + '  &  ' + '' + ' & ' + line
                print(line)

            if i_file < len(in_files)-1: print('\midrule')

        print(r'\bottomrule')
        print('\n')


def main(diff_type='ACD'):
    # diff_type = 'MCD'
    py_names = [
        'kmeans',
        'kmedian_l1',
        'kmedian',  # our method
        # # 'kmedian_tukey',
        'my_spectralclustering',
    ]
    init_methods = ['random', 'kmeans++', 'omniscient']
    in_dir = '/Users/kun/Downloads/rkm/20230312/4algorithms'
    if '4algorithms' in in_dir:
        cases = [
             # case 1: diff_outliers, mu+\- std/np.sqrt(n_repeats)
             # ([os.path.join(in_dir, f'Case1_R_1000_None_NONE_{diff_type}_-1th_1000_{init_method}_diff_outliers.csv')
             #   for init_method in init_methods],
             #  [2.0, 3.0, 4.0, 5.0, 6.0]),

            # case 2: diff2_outliers
            # ([os.path.join(in_dir, f'Case2_R_1000_None_NONE_{diff_type}_-1th_1000_{init_method}_diff2_outliers.csv')
            #   for init_method in init_methods],
            #  [0.5, 1.0, 2.0, 3.0, 4.0]),

            # # case 3: constructed2_3gaussians
            # ([os.path.join(in_dir, f'Case3_R_1000_None_NONE_{diff_type}_-1th_1000_{init_method}_constructed2_3gaussians.csv')
            #   for init_method in init_methods],
            #  [0.05, 0.10, 0.20, 0.35, 0.49]),

            # case 4: constructed_3gaussians
            # Case4_R_1000_None_NONE_MCD_- 1th_1000_random_constructed_3gaussians_4.csv
            ([os.path.join(in_dir, f'Case4_R_1000_None_NONE_{diff_type}_-1th_1000_{init_method}_constructed_3gaussians_4.csv')
              for init_method in init_methods],
             [0.05, 0.10, 0.20, 0.35, 0.49]),
        ]
    else:
        cases = [
        # case 4: constructed_3gaussians
        ([os.path.join(in_dir,
                       f'Case4_R_1000_{init_method}_NONE_{diff_type}_-1th_1000_{init_method}_constructed_3gaussians_3.csv')
          for init_method in init_methods],
         [0.05, 0.10, 0.20, 0.35, 0.49]),

        ]

    for case, ps in cases:
        print(diff_type, case, ps)
        _main(case, ps, py_names, precision=3, verbose=0)
        print('\n\n')


if __name__ == '__main__':
    main()

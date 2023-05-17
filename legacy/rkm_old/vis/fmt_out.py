import os.path

import pandas as pd


def fc(row):
    # print(row)
    st = set([str(_i) for _i in range(10)])

    precision = 1
    show_row = False
    for i, v in enumerate(row.values):
        # format each column value
        if type(v) != str:
            s = v
        else:
            s = ''
            cnt = 0
            flg = False
            for ch in v:
                if ch == '.':
                    # print(s)
                    if s[-1] in st and int(s[-1]) >= 5 and len(v) >= 100:
                        print(f'{i}th column: {v}', flush=True)
                        show_row = True
                    s += ch
                    flg = True
                    cnt = 0
                elif flg and ch in st and cnt < precision:
                    s += ch
                    cnt += 1
                elif cnt >= precision and ch in st:
                    flg = False
                else:
                    s += ch
                    cnt = 0
            # print(s)
        row[i] = s
    if show_row: print(f'{row.name}th row', flush=True)
    return row

def rowIndex(row):
    return row.name

def main():
    dim = 10
    dim = 200
    files = [
        # f'Case52_R_50_random_NONE_d:{dim}_r:0.10_mu:0,0_cov:5.0,5.0_ACD_-1th_3_3_50_random_gaussians10_ds_3.csv',
        # f'Case52_R_50_kmeans++_NONE_d/{dim}_r/0.10_mu/0,0_cov/5.0,5.0_ACD_-1th_3_3_50_kmeans++_gaussians10_ds_3.csv'.replace('/',':'),
        # f'Case52_R_50_omniscient_NONE_d/{dim}_r/0.10_mu/0,0_cov/5.0,5.0_ACD_-1th_3_3_50_omniscient_gaussians10_ds_3.csv'.replace('/',':')
        # 'Case52_R_50_random_NONE_d/200_r/0.10_mu/0,0_cov/5.0,5.0_ACD_-1th_3_3_50_random_gaussians10_ds_3.csv'.replace('/',':'),
         'Case52_R_50_kmeans++_NONE_d/200_r/0.10_mu/0,0_cov/5.0,5.0_ACD_-1th_3_3_50_kmeans++_gaussians10_ds_3.csv'.replace('/',':'),
         # 'Case52_R_50_omniscient_NONE_d/200_r/0.10_mu/0,0_cov/5.0,5.0_ACD_-1th_3_5_50_omniscient_gaussians10_ds_3.csv'.replace('/',':')
    ]
    in_dir = '/Users/kun/Downloads/rkm/20230502'
    files = [os.path.join(in_dir, v) for v in files]

    for in_file in files:
        df = pd.read_csv(in_file)
        df2 = df.apply(func=fc, axis=1)
        df2.to_csv(in_file + '-fmt.csv')


if __name__ == '__main__':
    main()

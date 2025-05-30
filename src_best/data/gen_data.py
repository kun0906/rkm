import math
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd


def letter_recognition(in_dir='data', n_clusters=3):
    """
        https://archive.ics.uci.edu/dataset/59/letter+recognition
        class: number of rows/points
        D    805
        P    803
        T    795
        M    792
        A    789
        X    787
        Y    786
        Q    783
        N    783
        F    775
        G    773
        E    768
        B    766
        V    764
        L    761
        R    758
        I    755
        O    753
        W    752
        S    748
        J    747
        K    739
        C    736
        H    734
        Z    734

    Parameters
    ----------
    in_dir

    Returns
    -------

    """
    in_file = os.path.join(in_dir, 'letter_recognition/letter-recognition.data')
    df = pd.read_csv(in_file, header=None)
    # print(df.iloc[:, 0].value_counts(sort=True))
    labels = ['A','F', 'C', ] # ['A', 'F', 'E']  # 'U' ['A', 'B', 'T']
    return df, labels[:n_clusters]


def biocoin_heist(in_dir='data', n_clusters=3):
    """
        https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset

        Top classes :   class               number of rows
            paduaCryptoWall                  12390
            montrealCryptoLocker              9315
            princetonCerber                   9223
            princetonLocky                    6625
            montrealCryptXXX                  2419
            montrealNoobCrypt                  483
            montrealDMALockerv3                354

    Returns
    -------

    """
    dat_file = os.path.join(in_dir, 'BitcoinHeistData.dat')
    if os.path.exists(dat_file):
        with open(dat_file, 'rb') as f:
            df = pickle.load(f)
    else:
        in_file = os.path.join(in_dir, 'BitcoinHeistData.csv')

        df = pd.read_csv(in_file)
        df = df.loc[:, ['label', 'length', 'weight', 'count', 'looped', 'neighbors', 'income']]
        # print(df['label'].value_counts(sort=True))
        with open(dat_file, 'wb') as f:
            pickle.dump(df, f)

    labels = ['paduaCryptoWall', 'montrealCryptoLocker', 'princetonLocky']

    return df, labels[:n_clusters]


def pen_digits(in_dir='data', n_clusters=3):
    """
        https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
        class: number of rows/points
        4    780
        0    780
        1    779
        7    778
        6    720
        5    720
        8    719
        9    719
        3    719

    Parameters
    ----------
    in_dir

    Returns
    -------

    """
    in_file = os.path.join(in_dir, 'pen_digits/pendigits.tra')
    df = pd.read_csv(in_file, header=None)
    d = df.shape[-1]
    df = df.iloc[:, [-1] + list(range(d - 1))]  # move the label to the first column
    # print(df['label'].value_counts(sort=True))
    # [0, 5, 7], [0, 5, 9], [0, 5, 2]
    labels = [0, 5, 2] # '0' [2, 4, 0] works for random, [2, 5, 7]
    return df, labels[:n_clusters]


def iot_intrusion(in_dir='data', n_clusters=3):
    """
        https://www.kaggle.com/datasets/subhajournal/iotintrusion
        class                       number of rows
        DDoS-UDP_Flood             121205
        DDoS-TCP_Flood             101293
        DDoS-PSHACK_Flood           92395
        DDoS-SYN_Flood              91644
        DDoS-RSTFINFlood            90823
        DDoS-SynonymousIP_Flood     80680
        DoS-UDP_Flood               74787
        DoS-TCP_Flood               59807
        DoS-SYN_Flood               45207
        BenignTraffic               24476
        Mirai-greeth_flood          22115
        Mirai-udpplain              20166
        Mirai-greip_flood           16952
        DDoS-ICMP_Fragmentation     10223
        MITM-ArpSpoofing             7019
        DDoS-ACK_Fragmentation       6431
        DDoS-UDP_Fragmentation       6431
        DNS_Spoofing                 4034
        Recon-HostDiscovery          3007
        Recon-OSScan                 2225
        Recon-PortScan               1863
        DoS-HTTP_Flood               1680
        VulnerabilityScan             809
        DDoS-HTTP_Flood               626
        DDoS-SlowLoris                493
        DictionaryBruteForce          324
        BrowserHijacking              140
        SqlInjection                  122
        CommandInjection              105
        Backdoor_Malware               76
        XSS                            72
        Recon-PingSweep                41
        Uploading_Attack               23

    Parameters
    ----------
    in_dir

    Returns
    -------

    """
    dat_file = os.path.join(in_dir, 'IoT_Intrusion.dat')
    if os.path.exists(dat_file):
        with open(dat_file, 'rb') as f:
            df = pickle.load(f)
    else:
        in_file = os.path.join(in_dir, 'IoT_Intrusion.csv')
        df = pd.read_csv(in_file, header=0)
        d = df.shape[-1]
        df = df.iloc[:, [-1] + list(range(d - 1))]  # move the label to the first column
        # print(df['label'].value_counts(sort=True))
        with open(dat_file, 'wb') as f:
            pickle.dump(df, f)
    labels = ['DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-SYN_Flood'] # ['DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-PSHACK_Flood']
    return df, labels[:n_clusters]


def cover_type(in_dir='data', n_clusters=3):
    """
        http://archive.ics.uci.edu/dataset/31/covertype
        class: number of rows/points
        1 211840
        2 283301
        3 35754
        4 2747
        5 9493
        6 17367
        7 20510
        (581012, 54)

    Parameters
    ----------
    in_dir

    Returns
    -------

    """
    in_file = os.path.join(in_dir, 'cover_type/covtype.data')
    df = pd.read_csv(in_file, header=None)
    d = df.shape[-1]
    df = df.iloc[:, [-1] + list(range(d - 1))]  # move the label to the first column
    # print(df['label'].value_counts(sort=True))
    tmp = df.iloc[:, 1:]
    normalized_df = (tmp - tmp.mean()) / tmp.std()
    df.iloc[:, 1:] = normalized_df.values
    labels = [1, 2, 3]
    return df, labels[:n_clusters]


def music_genre(in_dir='data', n_clusters=3):
    """
        https://www.kaggle.com/datasets/purumalgi/music-genre-classification?select=train.csv

        Index :
        10    4949
        6     2587
        9     2524
        8     1854
        5     1447
        1     1373
        2     1272
        0      625
        7      576
        3      402
        4      387
    Parameters
    ----------
    in_dir

    Returns
    -------

    """
    in_file = os.path.join(in_dir, 'music_genre/train.csv')
    df = pd.read_csv(in_file, header=0)
    # df['gender'] = df['gender'].replace({"M": 0, "F": 1})
    df = df.drop(columns=['Artist Name', 'Track Name'])
    df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
    d = df.shape[-1]
    df = df.iloc[:, [-1] + list(range(d - 1))]  # move the label to the first column
    # print(df['class'].value_counts(sort=True))
    labels = [3, 9, 10]  # [1, 4, 7], [1, 7, 10], [2, 6, 10], [3, 9, 10]
    return df, labels[:n_clusters]


def missing_stats(df):
    stat = df.isna().sum(axis=0).to_frame(name='missing').reset_index()  # sum() default is axis=0
    # https://stackoverflow.com/questions/17232013/how-to-set-the-pandas-dataframe-data-left-right-alignment
    df.style.set_properties(subset=['index'], **{'text-align': 'right'})
    n, d = df.shape
    stat['missing_percent'] = (stat['missing'] / n * 100).apply(lambda x: float(f'{x:.2f}'))
    stat['total'] = n
    missing_value_stat = stat.sort_values(by='missing_percent', ascending=False)
    # f = 'data/missing_value_stat.csv'
    # missing_value_stat.to_csv(f, sep=',')
    return missing_value_stat


def credit_loan(in_dir='data', n_clusters=3):
    """
         Credit risk

        dataset:
            https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

        Index :
        shape: (887379, 32)
        Current                                                601779
        Fully Paid                                             207723
        Charged Off                                             45248
        Late (31-120 days)                                      11591
        Issued                                                   8460
        In Grace Period                                          6253
        Late (16-30 days)                                        2357
        Does not meet the credit policy. Status:Fully Paid       1988
        Default                                                  1219
        Does not meet the credit policy. Status:Charged Off       761

        'Current', 'Fully Paid', 'Charged Off', 'Late (31-120 days)'
    Parameters
    ----------
    in_dir

    Returns
    -------

    """
    in_file = os.path.join(in_dir, 'credit_loan/loan.csv')

    df = pd.read_csv(in_file, header=0, sep=',')
    # TODO: Transform categorical features to numerical features
    df = df.drop(columns=[  # all categorical features
        'id', 'member_id', 'term', 'grade', 'sub_grade', 'emp_title', 'emp_length',
        'home_ownership', 'verification_status', 'issue_d', 'pymnt_plan',
        'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state',
        'earliest_cr_line', 'initial_list_status', 'last_pymnt_d', 'next_pymnt_d',
        'last_credit_pull_d', 'application_type', 'annual_inc_joint', 'dti_joint',
        'verification_status_joint',
        # too much missing valeus: > 51%
        'il_util', 'mths_since_rcnt_il', 'inq_last_12m', 'open_rv_12m', 'open_acc_6m',
        'open_il_6m', 'open_il_12m', 'open_il_24m', 'total_bal_il', 'max_bal_bc', 'all_util',
        'inq_fi', 'total_cu_tl', 'mths_since_last_record', 'mths_since_last_major_derog',
        'mths_since_last_delinq',
        # unique features
        'policy_code',

    ])
    # print(missing_stats(df))
    # print(f'shape: {df.shape}')
    # print(df['loan_status'].value_counts())
    # print(df.info())
    label = 'loan_status'
    # # df['gender'] = df['gender'].replace({"M": 0, "F": 1})
    df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
    # d = df.shape[-1]
    # print(df.describe())
    df = df.loc[:, [label] + [v for v in list(df.columns) if v != label]]  # move the label to the first column
    # print(df['class'].value_counts(sort=True))
    tmp = df.iloc[:, 1:]
    normalized_df = (tmp - tmp.mean()) / tmp.std()
    df.iloc[:, 1:] = normalized_df.values
    labels = ['Current', 'Fully Paid', 'Charged Off']
    # print(missing_stats(df))
    return df, labels[:n_clusters]


def plot_xy(X, Y, random_state=42, true_centroids = None, title=''):
    import matplotlib.pyplot as plt
    # PCA
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    print(X.shape)
    pca = PCA(n_components=2, svd_solver='full', random_state=random_state)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    # Y_true = Y.values
    X_embedded = pca.transform(X)
    centroids_embedded = pca.transform(true_centroids)
    # plt.rcParams["figure.figsize"] = (4,3)
    # colors = {'NT': 'r', 'OTG': 'g', 'RL': 'b'}
    colors = ["g", "b", "orange", "r", "m", 'black', 'brown', 'tab:green', 'tab:blue', 'tab:orange', 'tab:red']
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())
    print(len(markers), markers)
    for i, l in enumerate(sorted(set(Y))):
        # if i > 19: continue
        # if i > 6: continue
        # if i < 6 and i > 12: continue
        # if i < 12 and i > 18: continue
        mask = Y == l
        print(l, sum(mask))
        # axes.scatter(X_embedded[mask, 0], X_embedded[mask, 1], c=colors[i], label=l, alpha=0.5)
        axes.scatter(X_embedded[mask, 0], X_embedded[mask, 1], marker=markers[i], label=l, color=colors[i], alpha=0.2)
        # for idx, txt in enumerate(Y[mask]):
        #     axes.annotate(txt, (X_embedded[mask, 0][idx], X_embedded[mask, 1][idx]))
        axes.legend(loc="upper right", title="Label")
        if i >= len(centroids_embedded): break
        axes.scatter(centroids_embedded[i, 0], centroids_embedded[i, 1], marker='X', label=l, color=colors[i], alpha=1, s=200)
    # title = f"Data projected by PCA"
    axes.set_title(f'PCA {title}')
    plt.legend(title='Legend')
    # axes.set_axis_off()
    plt.show()

    # from sklearn.manifold import TSNE
    # print(X.shape)
    # perplexities = [2, 5, 15, 30, 50, 100]
    # fig, axes = plt.subplots(1, len(perplexities), figsize=(15, 3))
    # for j, perplexity in enumerate(perplexities):  # [2, 5, 15, 30, 50, 100]:
    #     X_embedded = TSNE(n_components=2,
    #                       init='pca', perplexity=perplexity, random_state=random_state).fit_transform(X)
    #     # plt.rcParams["figure.figsize"] = (4,3)
    #     # colors = {'NT': 'r', 'OTG': 'g', 'RL': 'b'}
    #     for idx, l in enumerate(sorted(set(Y))):
    #         mask = Y == l
    #         # axes[j].scatter(X_embedded[mask, 0], X_embedded[mask, 1], c=colors[idx], label=l, alpha=0.5)
    #         axes[j].scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=l, alpha=0.5)
    #     if j == 0:
    #         axes[j].legend(loc="upper right", title="Label")
    #     axes[j].set_title(f'perplexity:{perplexity}')
    # # plt.legend()
    # plt.title(f'PCA {title}')
    # plt.show()


def gen_data(data_name='letter_recognition', fake_label=True, n_clusters=4, each_cluster_size=100, prop=0.60,
             add_outlier=True, random_state=42):
    rng = np.random.RandomState(seed=random_state)
    # print(rng.choice(range(5), 4))
    # in_dir = 'data/letter_recognition'
    # in_file = os.path.join(in_dir, 'letter-recognition.data')
    # df = pd.read_csv(in_file)
    # labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if data_name == 'letter_recognition':
        df, labels = letter_recognition(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label == 'synthetic':
                outlier_std = 10

                # X = np.zeros((0, dim), dtype=float)
                # Y = np.zeros((0,), dtype=str)
                #
                # for i, l in enumerate(labels[:n_clusters]):
                #     tmp = df[df.iloc[:, 0] == l].values
                #     x, y = tmp[:, 1:], tmp[:, 0]
                #     x = x.astype('float')
                #     X = np.concatenate([X, x], axis=0)
                #     Y = np.concatenate([Y, y])

                mu = np.zeros(dim)
                # mu = np.mean(x, axis=0)

                outliers = rng.multivariate_normal(mu,
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))

            elif fake_label == 'OMC':
                # outliers from any class (including the inlier classes)
                outliers = df[~df.iloc[:, 0].isin(labels)].values  # true label with random feature values
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')  # only features without labels.
            elif fake_label == 'OOC': # J may work, R, D, H
                outliers = df[df.iloc[:, 0] == 'J'].values  # outliers from one class
                m, _ = outliers.shape
                # because prop changes, so rng.choice() will be effected to the latter results.
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
            else:
                raise ValueError('fake_label must be either "synthetic" or OMC or OOC')
        else:
            outliers = None

    elif data_name == "biocoin_heist":
        df, labels = biocoin_heist(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label=='synthetic':
                mu = np.zeros(dim)
                outlier_std = 5
                outliers = rng.multivariate_normal(mu,
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))

            elif fake_label == 'OMC':
                # outliers from any class (including the inlier classes)
                # outliers = df.values        # true label with random feature values
                outliers = df[~df.iloc[:, 0].isin(labels)].values
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')  # only features without labels.
            elif fake_label == 'OOC':
                outliers = df[df.iloc[:, 0] == 'montrealNoobCrypt'].values  # outliers from one class
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
        else:
            outliers = None

    elif data_name == "pen_digits":
        df, labels = pen_digits(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label== 'synthetic':
                outlier_std = 500
                mu = np.zeros(dim)
                # print(df.describe())
                # X = np.zeros((0, dim), dtype=float)
                # Y = np.zeros((0,), dtype=str)
                # for i, l in enumerate(labels[:n_clusters]):
                #     tmp = df[df.iloc[:, 0] == l].values
                #     x, y = tmp[:, 1:], tmp[:, 0]
                #     x = x.astype('float')
                #     X = np.concatenate([X, x], axis=0)
                #     Y = np.concatenate([Y, y])
                # mu = np.mean(X, axis=0)

                outliers = rng.multivariate_normal(mu,
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))
            elif fake_label == 'OMC':
                # outliers from any class (including the inlier classes)
                # outliers = df.values        # true label with random feature values
                outliers = df[~df.iloc[:, 0].isin(labels)].values
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')  # only features without labels.
            elif fake_label == 'OOC':
                outliers = df[df.iloc[:, 0] == 8].values  # outliers from one class
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
            else:
                raise ValueError('fake_label must be either "synthetic" or OMC or OOC')
        else:
            outliers = None

    elif data_name == "iot_intrusion":
        df, labels = iot_intrusion(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label=='synthetic':
                outlier_std = 3
                outliers = rng.multivariate_normal(np.zeros(dim),
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))
            elif fake_label=='random':
                # outliers from any class (including the inlier classes)
                # outliers = df.values        # true label with random feature values
                outliers = df[~df.iloc[:, 0].isin(labels)].values
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')  # only features without labels.
            else: # 'BenignTraffic', 'Mirai-greeth_flood'
                outliers = df[df.iloc[:, 0] == 'BenignTraffic'].values  # outliers from one class
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
        else:
            outliers = None

    elif data_name == "music_genre":
        df, labels = music_genre(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label == 'synthetic':
                outlier_std = 50
                mu = np.zeros(dim)

                # X = np.zeros((0, dim), dtype=float)
                # Y = np.zeros((0,), dtype=str)
                # for i, l in enumerate(labels[:n_clusters]):
                #     tmp = df[df.iloc[:, 0] == l].values
                #     x, y = tmp[:, 1:], tmp[:, 0]
                #     x = x.astype('float')
                #     X = np.concatenate([X, x], axis=0)
                #     Y = np.concatenate([Y, y])
                # mu = np.mean(X, axis=0)

                outliers = rng.multivariate_normal(mu,
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))
            elif fake_label == 'random':
                # outliers from any class (including the inlier classes)
                # outliers = df.values        # true label with random feature values
                outliers = df[~df.iloc[:, 0].isin(labels)].values
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')  # only features without labels.
            else:
                outliers = df[df.iloc[:, 0] == 7].values  # outliers from one class
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
        else:
            outliers = None

    elif data_name == "credit_loan":
        df, labels = credit_loan(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label=='synthetic':
                outlier_std = 10
                outliers = rng.multivariate_normal(np.zeros(dim),
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))
            elif fake_label == 'random':
                # outliers from any class (including the inlier classes)
                # outliers = df.values        # true label with random feature values
                outliers = df[~df.iloc[:, 0].isin(labels)].values
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')  # only features without labels.
            else:
                outliers = df[df.iloc[:, 0] == 0].values  # outliers from one class
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
        else:
            outliers = None

    elif data_name == "cover_type":
        df, labels = cover_type(n_clusters=n_clusters)
        dim = df.shape[-1] - 1
        if add_outlier:
            if fake_label:
                # # outliers from any class (including the inlier classes)
                # # outliers = df.values        # true label with random feature values
                # outliers = df[~df.iloc[:, 0].isin(labels)].values
                # m, _ = outliers.shape
                # indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                # outliers = outliers[indices, 1:].astype('float')  # only features without labels.

                outlier_std = 10
                outliers = rng.multivariate_normal(np.zeros(dim),
                                                   np.eye(dim) * outlier_std ** 2,
                                                   size=math.floor(each_cluster_size * prop))
            else:
                outliers = df[df.iloc[:, 0] == 0].values  # outliers from one class
                m, _ = outliers.shape
                indices = rng.choice(range(m), math.floor(each_cluster_size * prop))
                outliers = outliers[indices, 1:].astype('float')
        else:
            outliers = None
    else:
        raise ValueError(f'{data_name} is wrong.')

    # if n_clusters > len(labels):
    #     raise ValueError(f'{n_clusters} is too large')

    # data = df[df[0].isin()].values

    # plot_xy(df.iloc[:, 1:].values, df.iloc[:, 0].values, random_state=random_state,
    #         true_centroids=np.zeros((len(set(df.iloc[:, 0].values)), dim)))


    centroids = np.zeros((n_clusters, dim))

    X = np.zeros((0, dim), dtype=float)
    Y = np.zeros((0,), dtype=int)

    for i, l in enumerate(sorted(labels[:n_clusters])): # sorted by the labels
        tmp = df[df.iloc[:, 0] == l].values
        x, y = tmp[:, 1:], tmp[:, 0]
        x = x.astype('float')
        indices = rng.choice(range(len(y)), each_cluster_size)
        # print(random_state, i, l, indices)
        x, y = x[indices],  [i] * len(indices)
        centroids[i] = np.mean(x, axis=0)

        X = np.concatenate([X, x], axis=0)
        Y = np.concatenate([Y, y])
    # print(prop, rng, random_state, X, y)
    # plot_xy(np.concatenate([X, outliers], axis=0), np.concatenate([Y, np.asarray(['100']*len(outliers))], axis=0), random_state=random_state)
    return {'X': X, 'Y': Y, 'centroids': centroids, 'outliers': outliers}

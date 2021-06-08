import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from graspologic.plot import heatmap

parser = argparse.ArgumentParser()
# settings include:
# 1. `median`: median test statistic for various fixed K
# 2. `auto`: test stats box plots and heatmap for automatically chosen K 
parser.add_argument('setting', help='which plot to produce')
args = parser.parse_args()

def get_pair_result(result, xlab='distance', ylab='value'):
    """
    Return distance vs. test stats / p-values
    """
    num_samples = result.shape[0]
    pair_result = np.zeros((int(num_samples * (num_samples - 1) / 2), 2))
    count = 0
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pair_result[count, 0] = j - i
            pair_result[count, 1] = result[i, j]
            count += 1
    pair_result = pd.DataFrame(pair_result, columns=[xlab, ylab]).astype({xlab: 'int32'})
    return pair_result

if args.setting == 'median':
    xlab = 'Difference in Time'
    ylab = 'Median Test Statistic'
    klab = 'Number of Communities'
    fontsize = 20

    median_ts_all = []
    for K in range(5, 31):
        with open('outputs/enron_gcorrDC_teststats_unpooled_ZestimatedK{}_untransformed.pkl'.format(K), 'rb') as f:
            ts = pickle.load(f)
            ts_pair_result = get_pair_result(ts, xlab, ylab)
            median_ts = ts_pair_result.groupby(xlab).median()
            median_ts.loc[:, klab] = K
            median_ts_all.append(median_ts)
    median_ts_all = pd.concat(median_ts_all)

    plt.figure(figsize=(14, 10))
    sns.set(font_scale=2)
    sns.lineplot(data=median_ts_all, x=xlab, y=ylab, hue=klab)
    plt.title('Median Test Statistic on Enron Email Dataset with Various Number of Community', fontsize=fontsize)
    plt.savefig('figures/enron_teststats_median.png')


elif args.setting == 'auto':
    xlab = 'Difference in Time'
    ylab = 'Test Statistic'
    fontsize = 20

    with open('outputs/enron_gcorrDC_teststats_unpooled_Zestimated_untransformed_fixseed.pkl', 'rb') as f:
        ts = pickle.load(f)
    ts_pair_result = get_pair_result(ts, xlab, ylab)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    heatmap(ts, ax=axs[0])
    axs[0].set_xlabel('Time Point', fontsize=fontsize)
    axs[0].set_ylabel('Time Point', fontsize=fontsize)
    axs[0].set_title('Test Statistic (Number of Communities Chosen Automatically)', fontsize=fontsize)
    p = sns.boxplot(x=xlab, y=ylab, data=ts_pair_result, ax=axs[1])
    p.set_xlabel(xlab, fontsize=fontsize)
    p.set_ylabel(ylab, fontsize=fontsize)
    axs[1].set_title('Test Statistic vs. Difference in Time', fontsize=fontsize)
    plt.savefig('figures/enron_teststats_autoK.png')
"""
Apply transformation on all the graphs and save output
"""
import numpy as np
import pickle 
import argparse
from utils import binarize
from graspologic.utils import pass_to_ranks

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('transformation', help='transformation applied to the graphs')
args = parser.parse_args()

def apply_func_to_all_graphs(g, func, **kwargs):
    g_new = np.zeros(g.shape)
    for i in range(g.shape[0]):
        g_new[i, ...] = func(g[i, ...], **kwargs)
    return g_new


with open('data/{}_graphs_untransformed.pkl'.format(args.data), 'rb') as f:
    graphs_orig = pickle.load(f)


output_path = 'data/{}_graphs'.format(args.data)
if args.transformation == 'binarize':
    graphs = apply_func_to_all_graphs(graphs_orig, binarize)
    output_path += '_binarize'
elif args.transformation == 'simpleNonzero':
	graphs = apply_func_to_all_graphs(graphs_orig, pass_to_ranks)
	output_path += '_simpleNonzero'
else:
    raise ValueError('{} is not implemented'.format(args.transformation))
output_path += '.pkl'

with open(output_path, 'wb') as f:
	pickle.dump(graphs, f)
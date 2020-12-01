"""
Apply transformation on all the graphs and save output
"""
import numpy as np
import pickle 
import argparse
from utils import binarize

parser = argparse.ArgumentParser()
parser.add_argument('transformation', help='transformation applied to the graphs')
args = parser.parse_args()

def apply_func_to_all_graphs(g, func):
    g_new = np.zeros(g.shape)
    for i in range(g.shape[0]):
        g_new[i, ...] = func(g[i, ...])
    return g_new


with open('data/mouse_graphs.pkl', 'rb') as f:
    graphs_orig = pickle.load(f)


output_path = 'data/mouse_graphs'
if args.transformation == 'binarize':
    graphs = apply_func_to_all_graphs(graphs_orig, binarize)
    output_path += '_binarize'
else:
    raise ValueError('{} is not implemented'.format(args.transformation))
output_path += '.pkl'

with open(output_path, 'wb') as f:
	pickle.dump(graphs, f)
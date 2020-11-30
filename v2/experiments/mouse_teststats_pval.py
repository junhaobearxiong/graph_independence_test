import numpy as np
import pickle
import argparse
from core import gcorr, community_estimation, block_permutation, permutation_pvalue


parser = argparse.ArgumentParser()
parser.add_argument('option', type=int, help='1: test statistics; 2: p-value')
parser.add_argument('Z_given', type=int, nargs='?', default=1, help='whether the true community assignment is given')
parser.add_argument('num_iter', type=int, nargs='?', default=500, help='number of iterations of permutation for computing p-value')
args = parser.parse_args()

output_path = 'outputs/'
if args.option == 1:
    output_path += 'mouse_teststats'
elif args.option == 2:
    output_path += 'mouse_pvalues'

if args.Z_given:
    output_path += '_Zgiven'
else:
    output_path += '_Zestimated'
output_path += '.pkl'


with open('data/mouse_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
with open('data/mouse_community_assignments.pkl', 'rb') as f:
    Ztrue = pickle.load(f)
with open('data/mouse_Zhat_dict.pkl', 'rb') as f:
    Zhat = pickle.load(f)


def run_test_stats():
    num_graphs = graphs.shape[0]
    test_stats = np.zeros((num_graphs, num_graphs))

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print('{}, {}'.format(i, j))
            if args.Z_given:
                Z = Ztrue
            else:
                Z = Zhat[(i, j)]
            G1 = graphs[i, ...]
            G2 = graphs[j, ...]
            test_stats[i, j] = gcorr(G1, G2, Z)

    test_stats += test_stats.T
    test_stats[np.diag_indices_from(test_stats)] = 1
    return test_stats


def run_pvalue():
    num_graphs = graphs.shape[0]
    pvalues = np.zeros((num_graphs, num_graphs))

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print('{}, {}'.format(i, j))
            if args.Z_given:
                Z = Ztrue
            else:
                Z = Zhat[(i, j)]
            G1 = graphs[i, ...]
            G2 = graphs[j, ...]
            pvalues[i, j] = permutation_pvalue(G1, G2, Z, args.num_iter)

    pvalues += pvalues.T
    # TODO: if want to be really rigorous, might do permutation test for this as well
    # since theoretically null distribution could have test stats = 1
    pvalues[np.diag_indices_from(pvalues)] = 0
    return pvalues


if args.option == 1:
    results = run_test_stats()
elif args.option == 2:
    results = run_pvalue()

with open(output_path, 'wb') as f:
    pickle.dump(results, f)
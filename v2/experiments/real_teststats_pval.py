import numpy as np
import pickle
import argparse
from core import gcorr, community_estimation, block_permutation, permutation_pvalue, gcorr_dcsbm, dcsbm_pvalue
from utils import binarize

parser = argparse.ArgumentParser()
parser.add_argument('data', help='`mouse` or `timeseries`')
parser.add_argument('option', type=int, help='1: test statistics; 2: p-value')
parser.add_argument('dc', type=int, help='whether to use the degree corrected test statistics')
parser.add_argument('Z_given', type=int, nargs='?', default=0, help='whether the true community assignment is given')
parser.add_argument('transformation', nargs='?', default='untransformed', help='transformation applied to the graphs')
parser.add_argument('num_iter', type=int, nargs='?', default=500, help='number of iterations of permutation for computing p-value')
args = parser.parse_args()

# format output path
output_path = 'outputs/{}'.format(args.data)
if args.option == 1:
    output_path += '_teststats'
elif args.option == 2:
    output_path += '_pvalues'

if args.dc:
    # if doing the degree-corrected version, we use the community estimation implemented in `DCSBMEstimator`
    output_path += '_dc'
    args.Z_given = 0

if args.Z_given:
    output_path += '_Zgiven'
else:
    output_path += '_Zestimated'

# format input path
graphs_input_path = 'data/{}_graphs'.format(args.data)
Zhat_input_path = 'outputs/{}_Zhat_dict'.format(args.data)
if args.transformation not in [
        'untransformed',
        'binarize',
        'simpleNonzero'
    ]:
    raise ValueError('{} is not implemented'.format(args.transformation))
else :
    graphs_input_path += '_' + args.transformation
    Zhat_input_path += '_' + args.transformation
    output_path += '_' + args.transformation

with open(graphs_input_path + '.pkl', 'rb') as f:
    graphs = pickle.load(f)

if not args.dc:
    if args.Z_given:
        with open('data/{}_community_assignments.pkl'.format(args.data), 'rb') as f:
            Ztrue = pickle.load(f)
    else:
        with open(Zhat_input_path + '.pkl', 'rb') as f:
            Zhat = pickle.load(f)

# set the number of maximum components for DC-SBM community estimation
if args.data == 'mouse':
    max_comm = 20
elif args.data == 'timeseries':
    max_comm = 10

print('output path: {}'.format(output_path))

def run_test_stats():
    num_graphs = graphs.shape[0]
    test_stats = np.zeros((num_graphs, num_graphs))

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print('{}, {}'.format(i, j))
            G1 = graphs[i, ...]
            G2 = graphs[j, ...]
            if not args.dc:
                if args.Z_given:
                    Z = Ztrue
                else:
                    Z = Zhat[(i, j)]
                test_stats[i, j] = gcorr(G1, G2, Z)
            else:
                test_stats[i, j] = gcorr_dcsbm(G1, G2, max_comm=max_comm)

    test_stats += test_stats.T
    test_stats[np.diag_indices_from(test_stats)] = 1
    return test_stats


def run_pvalue():
    num_graphs = graphs.shape[0]
    pvalues = np.zeros((num_graphs, num_graphs))

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print('{}, {}'.format(i, j))

            G1 = graphs[i, ...]
            G2 = graphs[j, ...]
            if not args.dc:
                if args.Z_given:
                    Z = Ztrue
                else:
                    Z = Zhat[(i, j)]
                pvalues[i, j] = permutation_pvalue(G1, G2, Z=Z, num_perm=args.num_iter)
            else:
                pvalues[i, j] = dcsbm_pvalue(G1, G2, max_comm=max_comm, num_perm=args.num_iter)

    pvalues += pvalues.T
    # TODO: if want to be really rigorous, might do permutation test for this as well
    # since theoretically null distribution could have test stats = 1
    pvalues[np.diag_indices_from(pvalues)] = 0
    return pvalues


if args.option == 1:
    results = run_test_stats()
elif args.option == 2:
    results = run_pvalue()

with open(output_path + '.pkl', 'wb') as f:
    pickle.dump(results, f)
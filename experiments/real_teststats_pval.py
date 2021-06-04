import numpy as np
import pickle
import argparse
from core import gcorr, community_estimation, block_permutation, block_permutation_pvalue, gcorr_dcsbm, dcsbm_pvalue, pearson_graph
from utils import binarize

parser = argparse.ArgumentParser()
parser.add_argument('data', help='`mouse` or `timeseries` or `cpac200` or `enron`')
parser.add_argument('test', type=int, help='1: gcorr, 2: dcsbm gcorr, 3: pearson')
parser.add_argument('option', type=int, help='1: test statistics; 2: p-value')
parser.add_argument('pooled_variance', type=int, nargs='?', default=1, 
    help='whether to use the pooled variance test statistics')
parser.add_argument('Z', type=int, nargs='?', default=1, 
    help='1: given true assignments (not available for dcsbm gcorr, 2: used estimated assignments (with optimally chosen number of K), 3: estimate assignments with fixed K')
parser.add_argument('transformation', nargs='?', default='untransformed', help='transformation applied to the graphs')
parser.add_argument('num_iter', type=int, nargs='?', default=500, help='number of iterations of permutation for computing p-value')
args = parser.parse_args()


# set specific parameters for DC-SBM community estimation
epsilon1 = 1e-3
epsilon2 = 1e-3
min_comm = 1
if 'mouse' in args.data:
    max_comm = 20
elif 'timeseries' in args.data:
    max_comm = 10
elif 'cpac200' in args.data:
    max_comm = 10
elif 'enron' in args.data:
    max_comm = 30
    epsilon1 = 1e-5


def format_output_path(args):
    # format output path
    output_path = 'outputs/{}'.format(args.data)

    if args.test == 1:
        output_path += '_gcorr'
    elif args.test == 2:
        output_path += '_gcorrDC'
    elif args.test == 3:
        output_path += '_pearson'
    else:
        raise ValueError('option {} for `test` is not implemented'.format(args.test))

    if args.option == 1:
        output_path += '_teststats'
    elif args.option == 2:
        output_path += '_pvalues'

    if args.test != 3:
        if args.pooled_variance:
            output_path += '_pooled'
        else:
            output_path += '_unpooled'

        if args.Z == 1:
            output_path += '_Ztrue'
        elif args.Z == 2:
            output_path += '_Zestimated'
        # note in this case, the number of community = `max_comm`
        elif args.Z == 3:
            output_path += '_ZestimatedK{}'.format(max_comm)

    if args.transformation not in [
        'untransformed',
        'binarize',
        'simpleNonzero',
        'log10'
    ]:
        raise ValueError('{} is not implemented'.format(args.transformation))

    output_path += '_' + args.transformation

    if args.option == 2:
        output_path += '_numperm' + str(args.num_iter)

    return output_path


output_path = format_output_path(args)
print('output path: {}'.format(output_path))
print('-------------------------------------------------------------------------------------------')

# format input path
graphs_input_path = 'data/{}_graphs'.format(args.data)
Zhat_input_path = 'outputs/{}_Zhat_dict'.format(args.data)
graphs_input_path += '_' + args.transformation
Zhat_input_path += '_' + args.transformation

with open(graphs_input_path + '.pkl', 'rb') as f:
    graphs = pickle.load(f)

if args.test != 3 and args.Z == 1:
    with open('data/{}_community_assignments.pkl'.format(args.data), 'rb') as f:
        Ztrue = pickle.load(f)

# TODO: note `community_estimation` uses MASE, 
# while community estimation within `graspologic.DCSBMEstimator` uses a separate LSE for each graph
if args.test == 1 and args.Z == 2:
    with open(Zhat_input_path + '.pkl', 'rb') as f:
        Zhat = pickle.load(f)

def run_test_stats():
    num_graphs = graphs.shape[0]
    test_stats = np.zeros((num_graphs, num_graphs))
    if args.test == 2:
        dcsbm_fit = {}

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print('{}, {}'.format(i, j))
            G1 = graphs[i, ...]
            G2 = graphs[j, ...]
            if args.test == 1:
                if args.Z == 1:
                    Z = Ztrue
                elif args.Z == 2:
                    Z = Zhat[(i, j)]
                elif args.Z == 3:
                    Z = community_estimation(G1, G2, min_components=max_comm, max_components=max_comm)
                test_stats[i, j] = gcorr(G1, G2, Z, pooled_variance=args.pooled_variance)
            elif args.test == 2:
                Z = None
                if args.Z == 3:
                    min_comm = max_comm
                elif args.Z == 2:
                    min_comm = 1
                elif args.Z == 1:
                    min_comm = 1 # is NOT used
                    Z = Ztrue
                ts, fit = gcorr_dcsbm(G1, G2, min_comm=min_comm, max_comm=max_comm, epsilon1=epsilon1, epsilon2=epsilon2,
                    pooled_variance=args.pooled_variance, Z=Z, return_fit=True)
                test_stats[i, j] = ts
                dcsbm_fit[(i, j)] = fit
            elif args.test == 3:
                test_stats[i, j] = pearson_graph(G1, G2)

    if args.test == 2:
        return test_stats, dcsbm_fit
    else:
        return test_stats


def run_pvalue():
    num_graphs = graphs.shape[0]
    pvalues = np.zeros((num_graphs, num_graphs))

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print('{}, {}'.format(i, j))
            G1 = graphs[i, ...]
            G2 = graphs[j, ...]
            if args.test == 1:
                if args.Z == 1:
                    Z = Ztrue
                elif args.Z == 2:
                    Z = Zhat[(i, j)]
                elif args.Z == 3:
                    Z = community_estimation(G1, G2, min_components=max_comm, max_components=max_comm)
                pvalues[i, j] = block_permutation_pvalue(G1, G2, test='gcorr', num_perm=args.num_iter, Z=Z)
            elif args.test == 2:
                Z = None
                if args.Z == 3:
                    min_comm = max_comm
                elif args.Z == 2:
                    min_comm = 1
                elif args.Z == 1:
                    min_comm = 1 # is NOT used
                    Z = Ztrue
                pvalues[i, j] = dcsbm_pvalue(G1, G2, min_comm=min_comm, max_comm=max_comm, epsilon1=epsilon1, epsilon2=epsilon2,
                    pooled_variance=args.pooled_variance, num_perm=args.num_iter, Z=Z)
            elif args.test == 3:
                pvalues[i, j] = block_permutation_pvalue(G1, G2, test='pearson', num_perm=args.num_iter)

    return pvalues


if args.option == 1:
    if args.test == 2:
        results, fits = run_test_stats()
        with open(output_path + '_fits.pkl', 'wb') as f:
            pickle.dump(fits, f)
    else:
        results = run_test_stats()
elif args.option == 2:
    results = run_pvalue()

with open(output_path + '.pkl', 'wb') as f:
    pickle.dump(results, f)

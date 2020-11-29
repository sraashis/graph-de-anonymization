import argparse as ap
import itertools
import math
import multiprocessing as mp

import networkx as nx

import utils
from seed_based_main import seed_based_mapping


def compute_init_seed(i, N, m, n, d1, d2, g1, g2):
    print(f'{i}/{N} Done')
    g1_nbrs2 = utils.knbrs(g1, m, 2)
    g1_nbrs3 = utils.knbrs(g1, m, 3)

    g1_nbrs2_len = len(g1_nbrs2) - d1
    g1_nbrs3_len = len(g1_nbrs3) - d1 - g1_nbrs2_len

    g2_nbrs2 = utils.knbrs(g2, n, 2)
    g2_nbrs3 = utils.knbrs(g2, n, 3)

    g2_nbrs2_len = len(g2_nbrs2) - d2
    g2_nbrs3_len = len(g2_nbrs3) - d2 - g2_nbrs2_len

    r1 = abs((d1 * 2) / (d1 * (d1 - 1)) - (d2 * 2) / (
            d2 * (d2 - 1)))
    r2 = abs(g1_nbrs2_len / len(g1_nbrs2) - g2_nbrs2_len / len(g2_nbrs2))
    r3 = abs(g1_nbrs3_len / len(g1_nbrs3) - g2_nbrs3_len / len(g2_nbrs3))
    return (m, n), math.log(3 / (r1 + r2 + r3))


def seed_free_mapping(args):
    print(f'##### Running with {args["num_workers"]} workers... #####')

    g1 = nx.read_edgelist(args["g1_edgelist_file"], nodetype=int)
    g2 = nx.read_edgelist(args["g2_edgelist_file"], nodetype=int)

    lim = min(g1.number_of_nodes(), args['seed_init_num'], g2.number_of_nodes())

    deg1 = dict(g1.degree())
    deg1 = sorted(deg1.items(), key=lambda kv: kv[1], reverse=True)[0:lim]
    deg2 = dict(g2.degree())
    deg2 = sorted(deg2.items(), key=lambda kv: kv[1], reverse=True)[0:lim]

    params = []
    pairs = list(itertools.product(deg1, deg2))
    for i, ((m, d1), (n, d2)) in enumerate(pairs, 1):
        params.append([i, len(pairs), m, n, d1, d2, g1, g2])

    SEED = {}
    with mp.Pool(processes=args['num_workers']) as pool:
        for (m, n), s in pool.starmap_async(compute_init_seed, params).get():
            SEED[(m, n)] = s
    SEED = sorted(SEED.items(), key=lambda kv: kv[1], reverse=True)[0:args['seed_init_num']]
    seed_str = [f'{a} {b}\n' for (a, b), _ in SEED]

    with open(args["output_file"], 'w') as f:
        f.writelines(seed_str)
        f.flush()


if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument("-nw", "--num_workers", required=True, type=int, help="Number of workers.")
    ap.add_argument("-g1", "--g1_edgelist_file", required=True, type=str, help="Path to g1 edgelist.")
    ap.add_argument("-g2", "--g2_edgelist_file", required=True, type=str, help="Path to g2 edgelist.")
    ap.add_argument("-init", "--seed_init_num", default=500, type=int, help="Path to g1->g2 seed nodes mapping.")
    ap.add_argument("-out", "--output_file", default="mapping_result.txt", type=str, help="Path to output file.")
    ap.add_argument("-gi", "--map_per_itr", default=500, type=int,
                    help="Number of nodes to map on each global iteration")
    args = vars(ap.parse_args())
    seed_free_mapping(args)
    args['seed_mapping_file'] = args['output_file']
    seed_based_mapping(args)

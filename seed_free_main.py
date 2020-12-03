import argparse as ap
import math
import multiprocessing as mp

import networkx as nx

import utils
from seed_based_main import seed_based_mapping


def compute_init_seed(i1, N, m, d1, deg2, deg2_len, g1, g2):
    print(f'From g1 {i1}/{N}')

    g1_nbrs1 = utils.knbrs(g1, m, 1)
    g1_nbrs1_len = len(g1_nbrs1) + 1
    g1_nbrs1_edge_len = g1.subgraph(g1_nbrs1).number_of_edges()

    g1_nbrs2 = utils.knbrs(g1, m, 2)
    g1_nbrs2_len = len(g1_nbrs2)
    g1_nbrs2_only = set(g1_nbrs2) - set(g1_nbrs1)
    g1_nbrs2_edge_len = g1.subgraph(g1_nbrs2_only).number_of_edges()

    g1_nbrs3 = utils.knbrs(g1, m, 3)
    g1_nbrs3_len = len(g1_nbrs3)
    g1_nbrs3_only = set(g1_nbrs3) - set(g1_nbrs2)
    g1_nbrs3_edge_len = g1.subgraph(g1_nbrs3_only).number_of_edges()

    sim = {}
    for i2, (n, d2) in enumerate(deg2, 1):
        print(f'\tg1 {i1}/{N} to g2 {n}[{i2}/{deg2_len}]')
        g2_nbrs1 = utils.knbrs(g2, n, 1)
        g2_nbrs1_len = len(g2_nbrs1)
        g2_nbrs1_edge_len = g2.subgraph(g2_nbrs1).number_of_edges()

        r1_A = (g1_nbrs1_edge_len * g2_nbrs1_edge_len) / (g1_nbrs1_len * g2_nbrs1_len)
        r1_B = abs(d1 / g1_nbrs1_edge_len - d2 / g2_nbrs1_edge_len)
        r1 = r1_A * r1_B

        g2_nbrs2 = utils.knbrs(g2, n, 2)
        g2_nbrs2_len = len(g2_nbrs2)
        g2_nbrs2_only = set(g2_nbrs2) - set(g2_nbrs1)
        g2_nbrs2_edge_len = g2.subgraph(g2_nbrs2_only).number_of_edges()

        r2_A = (g1_nbrs2_edge_len * g2_nbrs2_edge_len) / (len(g1_nbrs2_only) * len(g2_nbrs2_only))
        r2_B = abs(len(g1_nbrs2_only) / g1_nbrs2_len - len(g2_nbrs2_only) / g2_nbrs2_len)
        r2 = r2_A * r2_B

        g2_nbrs3 = utils.knbrs(g2, n, 3)
        g2_nbrs3_len = len(g2_nbrs3)
        g2_nbrs3_only = set(g2_nbrs3) - set(g2_nbrs2)
        g2_nbrs3_edge_len = g2.subgraph(g2_nbrs3_only).number_of_edges()
        r3_A = (g1_nbrs3_edge_len * g2_nbrs3_edge_len) / (len(g1_nbrs3_only) * len(g2_nbrs3_only))
        r3_B = abs(len(g1_nbrs3_only) / g1_nbrs3_len - len(g2_nbrs3_only) / g2_nbrs3_len)

        r3 = r3_A * r3_B

        r = math.exp(3 / math.exp(r1 + r2 + r3)) - 1
        sim[(m, n)] = r

    top = max(sim, key=sim.get)
    return top, sim[top]


def seed_free_mapping(args):
    print(f'##### Running with {args["num_workers"]} workers... #####')

    g1 = nx.read_edgelist(args["g1_edgelist_file"], nodetype=int)
    g2 = nx.read_edgelist(args["g2_edgelist_file"], nodetype=int)

    lim = min(g1.number_of_nodes(), args['seed_init_num'], g2.number_of_nodes())

    deg1 = dict(g1.degree())
    deg1 = sorted(deg1.items(), key=lambda kv: kv[1], reverse=True)[0:lim]

    deg2 = dict(g2.degree())
    deg2 = sorted(deg2.items(), key=lambda kv: kv[1], reverse=True)[0:2 * lim]

    params = []
    for i, (m, d1) in enumerate(deg1, 1):
        params.append([i, lim, m, d1, deg2, len(deg2), g1, g2])

    SEED = {}
    with mp.Pool(processes=args['num_workers']) as pool:
        for (m, n), s in pool.starmap_async(compute_init_seed, params).get():
            SEED[(m, n)] = s
    seed_str = [f'{a} {b}\n' for (a, b), _ in SEED.items()]

    with open(args["output_file"], 'w') as f:
        f.writelines(seed_str)
        f.flush()


if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument("-nw", "--num_workers", required=True, type=int, help="Number of workers.")
    ap.add_argument("-g1", "--g1_edgelist_file", required=True, type=str, help="Path to g1 edgelist.")
    ap.add_argument("-g2", "--g2_edgelist_file", required=True, type=str, help="Path to g2 edgelist.")
    ap.add_argument("-sin", "--seed_init_num", default=500, type=int, help="Number of seeds mappings to extract.")
    ap.add_argument("-out", "--output_file", default="mapping_result.txt", type=str, help="Path to output file.")
    ap.add_argument("-mpi", "--map_per_itr", default=1000, type=int,
                    help="Number of nodes to map on each global iteration")
    args = vars(ap.parse_args())
    seed_free_mapping(args)
    args['seed_mapping_file'] = args['output_file']
    seed_based_mapping(args)

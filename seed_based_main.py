import networkx as nx

from utils import load_mapping, compute_mapping_with_seed
import multiprocessing as mp
import sys
import argparse as ap


def seed_based_mapping(args):
    print(f'##### Running with {args["num_workers"]} workers... #####')

    g1 = nx.read_edgelist(args["g1_edgelist_file"], nodetype=int)
    g2 = nx.read_edgelist(args["g2_edgelist_file"], nodetype=int)

    seed = load_mapping(args["seed_mapping_file"])
    MAPPING = {}.fromkeys(g1.nodes)
    MAPPING.update(seed)
    STRENGTH = {}.fromkeys(g1.nodes, 0.0)
    for n in seed:
        STRENGTH[n] = float('inf')

    ITR_LIM = (g1.number_of_nodes() - len(seed)) // args["map_per_itr"]
    ITR_LIM *= 2
    for i in range(1, ITR_LIM):
        g1_seed = set(seed.keys())
        g2_seed = set(seed.values())

        g1_nodes = set([k for k in MAPPING if MAPPING[k] is None])
        g1_len = len(g1_nodes)
        g2_nodes = set(g2.nodes) - set(MAPPING.values())
        g2_len = len(g2_nodes)
        _map = {}
        params = []
        for ix, m in enumerate(g1_nodes):
            params.append([i, ITR_LIM, ix, g1_len, g2_len, m, g2_nodes, seed, g1_seed, g2_seed, g1, g2])

        with mp.Pool(processes=args['num_workers']) as pool:
            for (m, n), s in pool.starmap_async(compute_mapping_with_seed, params).get():
                if m is not None and s > _map.get((m, n), 0):
                    _map[(m, n)] = s

        top_k = sorted(_map.items(), key=lambda kv: kv[1], reverse=True)[0:min(args["map_per_itr"], len(_map))]
        for (m, n), s in top_k:
            mapped = {m: s}
            for k in MAPPING:
                if MAPPING[k] == n:
                    mapped[k] = STRENGTH[k]

            a = max(mapped, key=mapped.get)
            MAPPING[a] = n
            seed[a] = n
            STRENGTH[a] = mapped[a]

            for b in set(mapped.keys()) - {a}:
                MAPPING[b] = None
                STRENGTH[b] = 0

        if None not in list(MAPPING.values()):
            break

        mappings_str = [f'{a} {b}\n' for a, b in list(MAPPING.items())]
        with open(args["output_file"], 'w') as f:
            f.writelines(mappings_str)
            f.flush()


if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument("-nw", "--num_workers", required=True, type=int, help="Number of workers.")
    ap.add_argument("-g1", "--g1_edgelist_file", required=True, type=str, help="Path to g1 edgelist.")
    ap.add_argument("-g2", "--g2_edgelist_file", required=True, type=str, help="Path to g2 edgelist.")
    ap.add_argument("-sm", "--seed_mapping_file", required=True, type=str, help="Path to g1->g2 seed nodes mapping.")
    ap.add_argument("-out", "--output_file", default="mapping_result.txt", type=str, help="Path to output file.")
    ap.add_argument("-mpi", "--map_per_itr", default=500, type=int,
                    help="Number of nodes to map on each global iteration")
    args = vars(ap.parse_args())
    seed_based_mapping(args)

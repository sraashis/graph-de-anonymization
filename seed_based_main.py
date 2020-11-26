import networkx as nx

from utils import load_mapping, compute_mapping_score
import multiprocessing as mp
import sys


if __name__ == "__main__":
    threads = int(sys.argv[1])
    print(f'Running with {threads} threads...')

    g1_edgelist_file = sys.argv[2]
    g2_edgelist_file = sys.argv[3]
    seed_mapping_file = sys.argv[4]
    out_file = sys.argv[5] if len(sys.argv) >= 6 else 'mapping_result.txt'
    g1 = nx.read_edgelist(g1_edgelist_file, nodetype=int)
    g2 = nx.read_edgelist(g2_edgelist_file, nodetype=int)

    seed = load_mapping(seed_mapping_file)
    seed_rev = dict([reversed(i) for i in seed.items()])
    g1_seed = set(seed.keys())
    g2_seed = set(seed.values())

    MAP_PER_IT = 1000
    MAPPING = {}.fromkeys(g1.nodes)
    MAPPING.update(seed)
    STRENGTH = {}.fromkeys(g1.nodes, 0.0)
    for n in seed:
        STRENGTH[n] = float('inf')
    ITR_LIM = (g1.number_of_nodes() - len(seed)) // MAP_PER_IT + 1

    for i in range(ITR_LIM):
        g1_nodes = set([k for k in MAPPING if MAPPING[k] is None])
        g1_len = len(g1_nodes)
        g2_nodes = set(g2.nodes) - set(MAPPING.values())
        g2_len = len(g2_nodes)
        _mapping = {}.fromkeys(g1_nodes)
        _strength = {}.fromkeys(g1_nodes, 0.0)

        params = []
        for ix, m in enumerate(g1_nodes):
            params.append([i, ITR_LIM, ix, g1_len, g2_len, m, g2_nodes, seed, g1_seed, g2_seed, g1, g2])

        with mp.Pool(processes=threads) as pool:
            for (m, n), s in pool.starmap_async(compute_mapping_score, params).get():
                if s > _strength[m]:
                    _mapping[m] = n
                    _strength[m] = s

        top_strength = sorted(_strength.items(), key=lambda kv: kv[1], reverse=True)[0:min(MAP_PER_IT, len(_strength))]

        for m, s in top_strength:
            if s > STRENGTH[m]:
                MAPPING[m] = _mapping[m]
                STRENGTH[m] = s

    mappings_str = [f'{a} {b}\n' for a, b in list(MAPPING.items())]
    f = open(out_file, 'w')
    f.writelines(mappings_str)
    f.flush()

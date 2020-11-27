import itertools
import math

import networkx as nx


def load_mapping(file):
    mapping = {}
    for u, v in ([n.strip().split(' ') for n in open(file).readlines()]):
        mapping[int(u)] = int(v)
    return mapping


def knbrs(g=None, start=None, k=1):
    knbrs = list(nx.single_source_shortest_path_length(g, start, cutoff=k).keys())
    knbrs.remove(start)
    return set(knbrs)


def gm(costs):
    w = 1 / len(costs)
    _gm = 0.0
    for c in costs:
        _gm += w * math.log(c + 1)
    return math.exp(_gm)


def get_seed_nbrs(g, n, k, seed):
    nbrs = set(knbrs(g, n, k))
    return nbrs, len(nbrs), nbrs.intersection(seed)


def compute_mapping_score(i, itr_lim, ix, g1_len, g2_len, m, g2_nodes, seed, g1_seed, g2_seed, g1, g2):
    print(f'ITR[{i}/{itr_lim}]: Matching node from g1 {m}[{ix}/{g1_len}]', end='\n')
    g1_nbrs1, g1_nbrs1_len, g1_nbrs1_seed = get_seed_nbrs(g1, m, 1, g1_seed)
    g1_nbrs2, g1_nbrs2_len, g1_nbrs2_seed = get_seed_nbrs(g1, m, 2, g1_seed)
    g1_nbrs3, g1_nbrs3_len, g1_nbrs3_seed = get_seed_nbrs(g1, m, 3, g1_seed)
    sim = {}
    for jx, n in enumerate(g2_nodes, 1):
        if jx % 500 == 0:
            print(f'\tg1 {m}[{ix}/{g1_len}] to g2: {n} [{jx}/{g2_len}]')

        c1, c2, c3 = 0, 0, 0
        m1, m2, m3 = 0, 0, 0
        g2_nbrs1, g2_nbrs1_len, g2_nbrs1_seed = get_seed_nbrs(g2, n, 1, g2_seed)

        for i, j in itertools.product(g1_nbrs1_seed, g2_nbrs1_seed):
            if seed[i] == j:
                m1 += 1

        c1 = m1 / (math.sqrt(g1_nbrs1_len) * math.sqrt(g2_nbrs1_len))

        if c1 > 0:
            g2_nbrs2, g2_nbrs2_len, g2_nbrs2_seed = get_seed_nbrs(g2, n, 2, g2_seed)

            for i, j in itertools.product(g1_nbrs2_seed, g2_nbrs2_seed):
                if seed[i] == j:
                    m2 += 1

            try:
                c2 = (m1 * m2) / (
                        math.log(g1_nbrs1_len * g2_nbrs1_len) *
                        math.sqrt(g1_nbrs2_len) * math.sqrt(g2_nbrs2_len))
            except:
                pass

        if c2 > 0:
            g2_nbrs3, g2_nbrs3_len, g2_nbrs3_seed = get_seed_nbrs(g2, n, 3, g2_seed)

            for i, j in itertools.product(g1_nbrs3_seed, g2_nbrs3_seed):
                if seed[i] == j:
                    m3 += 1
            try:
                c3 = (m1 * m2 * m3) / (
                            math.log(g1_nbrs1_len * g2_nbrs1_len * g1_nbrs2_len * g2_nbrs2_len) *
                            math.sqrt(g1_nbrs3_len) * math.sqrt(g2_nbrs3_len))
            except:
                pass

            sim[(m, n)] = round(c1 + c2 + c3, 6)

    if len(sim) > 0:
        top = max(sim, key=sim.get)
        strength = sim[top]
        print(f"### Matched: {top}, {strength}")
        return top, strength
    return (None, None), 0

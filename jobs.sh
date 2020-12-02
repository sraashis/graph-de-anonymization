#!/bin/bash
python seed_based_main.py -nw 32 -g1 seedbased/G1.edgelist -g2 seedbased/G2.edgelist -sm seedbased/seed_node_pairs.txt -out seedbased/seed_based_result.txt
python seed_free_main.py -nw 32 -g1 seedfree/G1.edgelist -g2 seedfree/G2.edgelist -out seedfree/seed_free_result.txt
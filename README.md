Implementation is available in the link [Github Link](https://github.com/sraashis/graph-de-anonymization) Report is [here](./Paper.pdf)


### Install libraries:
 - pip install -U numpy networkx tqdm torch pandas scikit-learn matplotlib


### ***Final results are in the output folder*** 


## Project No. 1: <hr>
### This is the implementation of graph de-anonymization in two ways:
- **Seed Based**   
    - It includes iterative seed propagation technique by matching multi-hop neighborhood structure
- **Seed Free**
    - It includes multi-hop density matching to extract confident seed and use the previous seed propagation algorithm.
    
### Usage for seed-based
    - python seed_based_main.py -nw 32 -g1 seedbased/G1.edgelist -g2 seedbased/G2.edgelist -sm seedbased/seed_node_pairs.txt -out seedbased/seed_based_result.txt
```commandline
  -nw NUM_WORKERS, --num_workers Number of workers/processes.
  -g1 G1_EDGELIST_FILE, --g1_edgelist_file, Path to g1 edgelist.
  -g2 G2_EDGELIST_FILE, --g2_edgelist_file, Path to g2 edgelist.
  -sm SEED_MAPPING_FILE, --seed_mapping_file, Path to g1->g2 seed nodes mapping.
  -out OUTPUT_FILE, --output_file,  Path to output file.
  -mpi MAP_PER_ITR, --map_per_itr, Number of nodes to map on each global iteration
```


### Usage for seed-free
    - python seed_free_main.py -nw 32 -g1 seedfree/G1.edgelist -g2 seedfree/G2.edgelist -out seedfree/seed_free_result.txt

 ```commandline

  -nw NUM_WORKERS, --num_workers, Number of workers/processes.
  -g1 G1_EDGELIST_FILE, --g1_edgelist_file, Path to g1 edgelist.
  -g2 G2_EDGELIST_FILE, --g2_edgelist_file, Path to g2 edgelist.
  -sin SEED_INIT_NUM, --seed_init_num, Number of seed mappings to extract.
  -out OUTPUT_FILE, --output_file,  Path to output file.
  -mpi MAP_PER_ITR, --map_per_itr, Number of nodes to map on each global iteration
```

### Full pipeline run
```
chmod x+u jobs.sh
./jobs.sh
```

<!-- ### Project No 2: <hr>
### The implementation is in AdultDataset.ipynb python notebook.
 -->

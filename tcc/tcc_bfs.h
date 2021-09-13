#ifndef _TCC_BFS_H_
#define _TCC_BFS_H_

#include "graph.h"

void tcc_bfs_do_bfs(graph* g, int root,
  int* parents, int* levels, 
  int**& level_queues, int* level_counts, int& num_levels);

void tcc_bfs_find_separators(graph* g, int* parents, int* levels, 
    int** level_queues, int* level_counts, int num_levels, 
    int* high, int* low, 
    int* tcc_labels, int* separators, int& num_separators);

void tcc_bfs(graph* g);

#endif

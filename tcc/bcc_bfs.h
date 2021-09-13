#ifndef __bcc_bfs_h__
#define __bcc_bfs_h__

#include "graph.h"

void bcc_bfs(graph* g, int* bcc_maps, int max_levels);

void bcc_bfs_do_bfs(graph* g, int root,
  int* parents, int* levels, 
  int**& level_queues, int* level_counts, int& num_levels);

void bcc_bfs_find_arts(graph* g, int* parents, int* levels, 
  int** level_queues, int* level_counts, int num_levels, 
  int* high, int* low, bool* art_point);

void bcc_bfs_do_final_bfs(graph* g, int root,
  int* parents, int* levels, 
  int** level_queues, int* level_counts, int num_levels, 
  int* high, int* low, bool* art_point);

void bcc_bfs_do_labeling(graph* g, int* labels, int& num_bcc, int& num_art,
                         bool* art_point, int* high, int* low, 
                         int* levels, int* parents);

#endif
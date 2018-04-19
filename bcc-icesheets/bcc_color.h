#ifndef __bcc_color_h__
#define __bcc_color_h__

#include "graph.h"

void bcc_color(graph* g, int* bcc_maps);

void bcc_color_do_bfs(graph* g, int root, 
  int* parents, int* levels);

int bcc_mutual_parent(graph* g, int* levels, int* parents, int vert, int out);

void bcc_color_init_mutual_parents(graph* g, int* parents, int* levels, 
  int* high, int* low);

void bcc_color_do_coloring(graph* g, int root,
  int* parents, int* levels,
  int* high, int* low);

void bcc_color_do_labeling(graph* g, int root, 
  int* labels, int& num_bcc, int& num_art,
  int* high, int* low, 
  int* levels, int* parents);

#endif
#ifndef _REDUCE_H_
#define _REDUCE_H_

#include "graph.h"

#define BLOCK_SIZE 512

#define ALPHA 15
#define BETA 24

template <typename T> void swap(T* &p, T* &q);

void add_new_edges(int* parents, int* new_srcs, int* new_dsts, 
                   int &new_num_edges, int num_verts);

int reduce_graph_gpu(graph* g, int* new_srcs, int* new_dsts, int& new_num_edges);

#endif

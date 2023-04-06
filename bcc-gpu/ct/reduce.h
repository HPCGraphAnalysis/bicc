#ifndef _REDUCE_H_
#define _REDUCE_H_

#include "graph.h"

#define DAMPING_FACTOR 0.85
#define BLOCK_SIZE 512

void add_new_edges(int* parents, int* new_srcs, int* new_dsts, 
                   int &new_num_edges, int num_verts);

void add_new_edges_union(int* f_parents, int* t_parents, int* new_srcs, 
                        int* new_dsts, int &new_num_edges, int num_verts);

template <typename T> void swap(T* &p, T* &q);

int reduce_graph_gpu(graph* g, int* new_srcs, int* new_dsts, int& new_num_edges);

#endif

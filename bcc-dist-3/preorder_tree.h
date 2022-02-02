#ifndef _PREORDER_TREE_H_
#define _PREORDER_TREE_H_

#include "dist_graph.h"
#include "comms.h"

int preorder_tree(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
  uint64_t* parents, uint64_t* levels, 
  uint64_t* preorder_label, uint64_t* num_descendents);

#endif

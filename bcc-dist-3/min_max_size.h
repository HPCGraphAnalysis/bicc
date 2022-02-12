#ifndef _MIN_MAX_SIZE_
#define _MIN_MAX_SIZE_

#include <stdint.h>

#include "dist_graph.h"
#include "comms.h"


int get_min_max_size(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
  uint64_t* parents, uint64_t* preorders, 
  uint64_t* num_descendents, uint64_t* mins, uint64_t* maxes);

#endif

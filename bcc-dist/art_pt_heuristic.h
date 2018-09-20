#ifndef __ART_PT_HEURISTIC_H__
#define __ART_PT_HEURISTIC_H__
#include "bicc_dist.h"
#include "dist_graph.h"
#include "comms.h"
void art_pt_heuristic(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, 
                     uint64_t* parents, uint64_t* levels, 
                     uint64_t* art_pt_flags);
#endif

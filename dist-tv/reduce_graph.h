#ifndef _REDUCE_GRAPH_H_
#define _REDUCE_GRAPH_H_

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "dist_graph.h"
#include "comms.h"
#include "util.h"
#include "bfs.h"

dist_graph_t* reduce_graph(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q);

int bicc_bfs_pull_reduced(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                   uint64_t* parents_old, uint64_t* parents,
                   uint64_t* levels, uint64_t root);

int spanning_tree(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                   uint64_t* parents, uint64_t* levels, uint64_t root);

int connected_components(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                         uint64_t* parents, uint64_t* labels);

int spanning_forest(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                    uint64_t* parents_old, uint64_t* parents, uint64_t* levels,
                    uint64_t* labels);


#endif

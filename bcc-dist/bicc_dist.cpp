/*
//@HEADER
// *****************************************************************************
//
//  XtraPuLP: Xtreme-Scale Graph Partitioning using Label Propagation
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/

#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <cstring>
#include <sys/time.h>
#include <time.h>

#include "bicc_dist.h"

#include "comms.h"
#include "dist_graph.h"
#include "bfs.h"
#include "lca.h"
#include "art_pt_heuristic.h"
//#include "color-bicc.h"
//#include "label.h"

int procid, nprocs;
int seed;
bool verbose, debug, verify;

extern "C" int bicc_dist_run(dist_graph_t* g)
{
  mpi_data_t comm;
  queue_data_t q;
  init_comm_data(&comm);
  init_queue_data(g, &q);

  bicc_dist(g, &comm, &q);

  clear_comm_data(&comm);
  clear_queue_data(&q);

  return 0;
}

extern "C" int bicc_dist(dist_graph_t* g,mpi_data_t* comm, queue_data_t* q)
{

  double elt = 0.0, elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

  uint64_t* parents = new uint64_t[g->n_total];
  uint64_t* levels = new uint64_t[g->n_total];
  bicc_bfs_pull(g, comm, q, parents, levels, g->max_degree_vert);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  for(int i = 0; i < g->n_local; i++){
    int curr_global = g->local_unmap[i];
    if(parents[i] != -1) printf("vertex %d, parent: %d, level: %d\n",curr_global, parents[i], levels[i]);
  }
  for(int i = 0; i < g->n_ghost; i++){
    int curr = g->n_local + i;
    printf("vertex %d, parent: %d, level: %d\n",g->ghost_unmap[i], parents[curr], levels[curr]);
  }
  
  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing BCC-LCA stage\n");
  }

  uint64_t* highs = new uint64_t[g->n_total];
  uint64_t* high_levels = new uint64_t[g->n_total];
  uint64_t* lows = new uint64_t[g->n_total];

  //for (int i = 0; i < 4; i++){
  //  printf("vertex %d, highs: %d, high_levels: %d\n",i,highs[i],high_levels[i]);
  //}

  //bicc_lca(g, comm, q, parents, levels, highs, high_levels);
  art_pt_heuristic(g,comm,q,parents,levels,highs);
  
  //TEST mark one edge as visited, one direction, to check that the array modifications work
  //uint64_t out_degree = out_degree(g,0);
  //if(out_degree > 1){
    //uint64_t* edge_visited = edge_visited(g, 0);
    //edge_visited[0] = 1;
  //}
  
  //check what edges were marked as visited
  //for(int i = 0; i < g->n_total; i++){
    //uint64_t out_degree = out_degree(g, i);
    //uint64_t* outs = out_vertices(g,i);
    //uint64_t* edge_visited = edge_visited(g, i);
    //for(int j = 0; j < out_degree; j++){
      //if(edge_visited[j] == 1){
      //  printf("edge from %d to %u was visited\n", i, outs[j]);
      //}
    //}
  //} 

 
  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    //elt = timer();
    //printf("Doing BCC-LCA stage\n");
  }

  delete [] parents;
  delete [] levels;
  delete [] high_levels;
  delete [] highs;
  delete [] lows;

  return 0;
}

extern "C" int create_dist_graph(dist_graph_t* g, 
          unsigned long n_global, unsigned long m_global, 
          unsigned long n_local, unsigned long m_local,
          unsigned long* local_adjs, unsigned long* local_offsets, 
          unsigned long* global_ids, unsigned long* vert_dist)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs > 1)
  {
    create_graph(g, (uint64_t)n_global, (uint64_t)m_global, 
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs, 
                 (uint64_t*)global_ids);
    relabel_edges(g, vert_dist);
  }
  else
  {
    create_graph_serial(g, (uint64_t)n_global, (uint64_t)m_global, 
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs);
  }

  //get_ghost_degrees(g);
  //get_ghost_weights(g);

  return 0;
}


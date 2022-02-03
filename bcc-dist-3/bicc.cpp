/*
//@HEADER
// *****************************************************************************
//
//  HPCGraph: Graph Computation on High Performance Computing Systems
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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "dist_graph.h"
#include "comms.h"
#include "util.h"
#include "bicc.h"

#define NOT_VISITED 18446744073709551615U
#define VISITED 18446744073709551614U
#define ASYNCH 1

extern int procid, nprocs;
extern bool verbose, debug, verify;

int bicc_bfs(dist_graph_t* g, mpi_data_t* comm, 
  uint64_t* parents, uint64_t* levels, uint64_t root)
{  
  if (debug) { printf("procid %d bicc_bfs() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  queue_data_t* q = (queue_data_t*)malloc(sizeof(queue_data_t));;
  init_queue_data(g, q);

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  uint64_t root_index = get_value(g->map, root);
  if (root_index != NULL_KEY && root_index < g->n_local)    
  {
    q->queue[0] = root;
    q->queue[1] = root;
    q->queue_size = 2;
  }
  
  bool* visited = new bool[g->n_total];

  uint64_t level = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  init_thread_queue(&tq);

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    parents[i] = NOT_VISITED;

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    levels[i] = NOT_VISITED;
  
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    visited[i] = false;
  
  while (comm->global_queue_size)
  {
#pragma omp single
    if (debug) { 
      printf("Task: %d bicc_bfs() GQ: %lu, TQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; i += 2)
    {
      uint64_t vert = q->queue[i];
      uint64_t parent = q->queue[i+1];
      uint64_t vert_index = get_value(g->map, vert);
      if (parents[vert_index] != VISITED && parents[vert_index] != NOT_VISITED)
        continue;
      
      parents[vert_index] = parent;
      levels[vert_index] = level;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);

      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
   
        uint64_t test = 0;
#pragma omp atomic capture
        { test = parents[out_index]; parents[out_index] = VISITED; }
        
        if (test == NOT_VISITED) {
          if (out_index < g->n_local) {
            add_vid_to_queue(&tq, q, g->local_unmap[out_index], vert);
          } else {
            add_vid_to_send(&tq, q, out_index, vert);
          }
        } else if (test != VISITED) {
          parents[out_index] = test;
        }
      }
    }  

    empty_queue(&tq, q);
    empty_send(&tq, q);
#pragma omp barrier

#pragma omp single
    {
      exchange_verts_bicc(g, comm, q);
      ++level;
    }
  } // end while

#pragma omp for nowait
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    uint64_t vert = g->local_unmap[vert_index];
    uint64_t parent = parents[vert_index];
    uint64_t parent_index = get_value(g->map, parent);
    if (parent_index >= g->n_local) {
      add_vid_to_send(&tq, q, parent_index, vert);
    }
  }
  
  empty_queue(&tq, q);
  empty_send(&tq, q);
#pragma omp barrier

#pragma omp single
{
  exchange_verts_bicc(g, comm, q);
}

#pragma omp for
  for (uint64_t i = 0; i < q->queue_size; i += 2) {
    uint64_t parent = q->queue[i];
    uint64_t child = q->queue[i+1];
    uint64_t child_index = get_value(g->map, child);
    parents[child_index] = parent;
  }

  clear_thread_queue(&tq);
} // end parallel

  clear_queue_data(q);
  free(q);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d bicc_bfs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d bicc_bfs() success\n", procid); }

  return 0;
}

int bicc_init_data(dist_graph_t* g, 
  uint64_t* parents, uint64_t* levels, 
  uint64_t* high, uint64_t* low, uint64_t* high_levels)
{
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    low[i] = g->local_unmap[i];
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    high[i] = parents[i];
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    high_levels[i] = levels[i] == 0 ? levels[i] : levels[i] - 1;

  return 0;
}


int run_bicc(dist_graph_t* g, mpi_data_t* comm, 
  uint64_t* high, uint64_t* low, uint64_t* parents, 
  uint64_t* levels, uint64_t* high_levels)
{ 
  if (debug) { printf("Task %d run_bicc() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t color_changes = g->n;
  uint64_t iter = 0;

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

#pragma omp parallel default(shared)
{
  thread_comm_t tc;
  init_thread_comm(&tc);
  
#pragma omp for schedule(guided)
  for (uint64_t i = 0; i < g->n_local; ++i) {
    update_sendcounts_thread(g, &tc, i, 3);
  }

  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic
    comm->sendcounts_temp[i] += tc.sendcounts_thread[i];

    tc.sendcounts_thread[i] = 0;
  }
#pragma omp barrier

#pragma omp single
{
  init_sendbuf_vid_data(comm);    
  init_recvbuf_vid_data(comm);
}

#pragma omp for schedule(guided)
  for (uint64_t i = 0; i < g->n_local; ++i) {
    update_vid_data_queues(g, &tc, comm, i, high[i], low[i], high_levels[i]);
  }

  empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
  exchange_verts(comm);
  exchange_data(comm);
}

#pragma omp for
  for (uint64_t i = 0; i < comm->total_recv; i += 3)
  {
    uint64_t vert_index = get_value(g->map, comm->recvbuf_vert[i]);
    assert(vert_index < g->n_total);
    high[vert_index] = comm->recvbuf_data[i];
    low[vert_index] = comm->recvbuf_data[i+1];
    high_levels[vert_index] = comm->recvbuf_data[i+2];
    comm->recvbuf_vert[i] = vert_index;
  }
  
#pragma omp single
  if (debug) printf("Task %d initialize ghost data success\n", procid);

#pragma omp for
  for (uint64_t i = 0; i < comm->total_send; ++i)
  {
    uint64_t index = get_value(g->map, comm->sendbuf_vert[i]);
    assert(index < g->n_total);
    comm->sendbuf_vert[i] = index;
  } 

#pragma omp single
  if (debug) printf("Task %d initialize send buff index success\n", procid);

  while (color_changes)
  {    
#pragma omp single
    if (debug) {
      printf("Task %d Iter %lu Changes %lu run_bicc(), %9.6lf\n", 
        procid, iter, color_changes, omp_get_wtime() - elt);
    }
  
#pragma omp single
    color_changes = 0;

#pragma omp for schedule(guided) reduction(+:color_changes)
    for (uint64_t i = 0; i < g->n_local; ++i)
    {
      uint64_t vert_index = i;
      uint64_t vert_high = high[vert_index];
      
      int vert = vert_index;
      uint64_t vert_high_level = high_levels[vert_index];
      uint64_t vert_low = low[vert_index];

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        uint64_t out;
        if (out_index < g->n_local)
          out = g->local_unmap[out_index];
        else
          out = g->ghost_unmap[out_index-g->n_local];
        uint64_t out_high = high[out_index];
        uint64_t out_high_level = high_levels[out_index];
        if (vert_high == out)
          continue;

        if (out_high_level == vert_high_level && out_high > vert_high)
        {
          high[vert_index] = out_high;
          vert_high = out_high;
          high_levels[vert_index] = out_high_level;
          vert_high_level = out_high_level;
          ++color_changes;
        }
        if (out_high_level < vert_high_level)
        {
          high[vert_index] = out_high;
          vert_high = out_high;
          high_levels[vert_index] = out_high_level;
          vert_high_level = out_high_level;
          ++color_changes;   
        }
        if (vert_high == out_high)
        {
          uint64_t out_low = low[out_index];
          if (out_low < vert_low)
          {
            low[vert_index] = out_low;
            vert_low = out_low;
            ++color_changes;
          }
        }
      }
    }

#pragma omp for
    for (uint64_t i = 0; i < comm->total_send; i += 3) {
      comm->sendbuf_data[i] = high[comm->sendbuf_vert[i]];
      comm->sendbuf_data[i+1] = low[comm->sendbuf_vert[i]];
      comm->sendbuf_data[i+2] = high_levels[comm->sendbuf_vert[i]];
    }

#pragma omp single
{
    exchange_data(comm);
}

#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; i += 3) {
      high[comm->recvbuf_vert[i]] = comm->recvbuf_data[i];
      low[comm->recvbuf_vert[i]] = comm->recvbuf_data[i+1];
      high_levels[comm->recvbuf_vert[i]] = comm->recvbuf_data[i+2];
    }

#pragma omp single
{
  MPI_Allreduce(MPI_IN_PLACE, &color_changes, 1, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  ++iter;
}
  } // end for loop

  clear_thread_comm(&tc);
} // end parallel

  clear_allbuf_vid_data(comm);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, run_bicc() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d run_bicc() success\n", procid); }

  return 0;
}

int run_art_pts(dist_graph_t* g, mpi_data_t* comm, uint64_t* high)
{
  if (debug) printf("Task %d run_art_pts() start\n", procid);

  bool* is_art = (bool*)malloc(g->n*sizeof(bool));
  int* art_points = new int[g->n];

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    is_art[i] = false;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i) {
    is_art[high[i]] = true;
    if (high[i] > g->n) {
      printf("too high %d %lu %lu\n", procid, i, high[i]);
      printf("%lu %lu\n", g->local_unmap[i], out_degree(g, i));
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, is_art, g->n, 
    MPI::BOOL, MPI_LOR, MPI_COMM_WORLD);

  uint64_t art_count = 0;
  for (uint64_t i = 0; i < g->n; ++i)
    if (is_art[i])
      art_points[art_count++] = i;

  if (procid == 0)
    printf("Articulation points: %lu\n", art_count);

  free(is_art);

  FILE* arts = fopen("arts", "w");
  for (int i = 0; i < art_count; ++i)
    fprintf(arts, "%d\n", art_points[i]);
  fclose(arts);
  
  if (debug) printf("Task %d run_art_pts() success\n", procid); 

  return 0;
}


int bicc_dist(dist_graph_t* g, mpi_data_t* comm, uint64_t root)
{  
  if (debug) printf("Task %d bicc_dist() start\n", procid);

  MPI_Barrier(MPI_COMM_WORLD);
  double elt = omp_get_wtime();

  uint64_t* parents = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  uint64_t* levels = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  root = 0;
  bicc_bfs(g, comm, parents, levels, root);

  uint64_t* high = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  uint64_t* low = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  uint64_t* high_levels = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  bicc_init_data(g, parents, levels, high, low, high_levels);

  run_bicc(g, comm, high, low, parents, levels, high_levels);
  free(low);
  free(high_levels);
  free(parents);
  free(levels);

  run_art_pts(g, comm, high);
  free(high);

  MPI_Barrier(MPI_COMM_WORLD);
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("BiCC time %9.6f (s)\n", elt);

  if (debug)  printf("Task %d bicc_dist() success\n", procid); 

  return 0;
}


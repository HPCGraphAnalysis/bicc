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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "dist_graph.h"
#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;


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

  // do full boundary exchange of parents data
  thread_comm_t tc;
  init_thread_comm(&tc);
  
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    add_vid_to_send(&tq, q, i);
    add_vid_to_queue(&tq, q, i);
  }

  empty_send(&tq, q);
  empty_queue(&tq, q);
#pragma omp barrier

  for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_sendcounts_thread(g, &tc, vert_index);
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
}

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_vid_data_queues(g, &tc, comm,
                           vert_index, parents[vert_index]);
  }

  empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
  exchange_vert_data(g, comm, q);
} // end single

#pragma omp for
  for (uint64_t i = 0; i < comm->total_recv; ++i)
  {
    uint64_t index = get_value(g->map, comm->recvbuf_vert[i]);
    parents[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_comm(&tc);
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


int bicc_bfs_pull(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                   uint64_t* parents, uint64_t* levels, uint64_t root)
{
  if (debug) { printf("procid %d bicc_bfs_pull() start\n", procid); }
  
  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  comm->global_queue_size = 1;
  uint64_t temp_send_size = 0;
  uint64_t level = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    parents[i] = NULL_KEY;
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    levels[i] = NULL_KEY;

#pragma omp single
{
  uint64_t root_index = get_value(g->map, root);
  if (root_index != NULL_KEY)    
  {
    parents[root_index] = root;
    levels[root_index] = 0;
  }
}

  while (comm->global_queue_size)
  {
    tq.thread_queue_size = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      if (parents[vert_index] != NULL_KEY)
        continue;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        if (levels[out_index] == level-1) {
          if (out_index < g->n_local)
            parents[vert_index] = g->local_unmap[out_index];
          else
            parents[vert_index] = g->ghost_unmap[out_index - g->n_local];
          levels[vert_index] = level;

          add_vid_to_send(&tq, q, vert_index);
          //add_vid_to_queue(&tq, q, vert_index);
          ++tq.thread_queue_size;
          break;
        }
      }
    }  

    empty_send(&tq, q);
    //empty_queue(&tq, q);

#pragma omp atomic
    q->next_size += tq.thread_queue_size;

#pragma omp barrier

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_sendcounts_thread(g, &tc, vert_index);
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
}

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_vid_data_queues(g, &tc, comm,
                             vert_index, parents[vert_index]);
    }

    empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
    temp_send_size = q->send_size;
    exchange_vert_data(g, comm, q);
} // end single


#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
    {
      uint64_t vert_index = get_value(g->map, comm->recvbuf_vert[i]);
      parents[vert_index] = comm->recvbuf_data[i];
      levels[vert_index] = level;
    }

#pragma omp single
{   
    clear_recvbuf_vid_data(comm);
    ++level;

    if (debug) printf("Rank %d send_size %lu global_size %li\n", 
      procid, temp_send_size, comm->global_queue_size);
}

  } // end while
} // end parallel 

  if (verify) {
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      if (parents[vert_index] != NULL_KEY) {
        if (levels[get_value(g->map, parents[vert_index])] != levels[vert_index] - 1)
          printf("Mismatch p: %lu pl: %lu - v: %lu vl: %lu\n",
            parents[vert_index], levels[get_value(g->map, parents[vert_index])],
            vert_index, levels[vert_index]);
      }
    }
  }

  if (debug) printf("Rank %d bicc_bfs_pull() success\n", procid);

  return 0;
}

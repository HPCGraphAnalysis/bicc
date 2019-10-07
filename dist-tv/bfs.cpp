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


int bicc_bfs(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
             uint64_t* parents, uint64_t* levels, bool* is_leaf, uint64_t root)
{  
  if (debug) { printf("procid %d wcc_bfs() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  uint64_t level = 0;
  comm->global_queue_size = 1;
  uint64_t temp_send_size = 0;
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
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    is_leaf[i] = true;

#pragma omp single
{
  uint64_t root_index = get_value(g->map, root);
  if (root_index != NULL_KEY && root_index < g->n_local)    
  {
    q->queue_next[0] = root;
    q->queue_size = 1;
    parents[root_index] = root;
  }
}
  while (comm->global_queue_size)
  {
    if (debug && tq.tid == 0) { 
      printf("Task: %d bicc_bfs() GQ: %lu, LQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert = q->queue_next[i];
      uint64_t vert_index = get_value(g->map, vert);
      // TODO: find out why vert = 0, is ending up being processed on
      //  tasks other then task 0 and remove send part of conditional
      //  below
      if (levels[vert_index] != NULL_KEY || vert_index > g->n_local)
        continue;
      levels[vert_index] = level;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        if (parents[out_index] == NULL_KEY)
        {
          is_leaf[vert_index] = false;
          parents[out_index] = vert;

          if (out_index < g->n_local)
            add_vid_to_queue(&tq, q, g->local_unmap[out_index]);
          else
            add_vid_to_send(&tq, q, out_index);
        }
      }
    }  

    empty_queue(&tq, q);
    empty_send(&tq, q);
#pragma omp barrier

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_sendcounts_thread_ghost(g, &tc, vert_index);
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
      update_vid_data_queues_ghost(g, &tc, comm,
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
      if (parents[vert_index] == NULL_KEY) {
        parents[vert_index] = comm->recvbuf_data[i];
        add_vid_to_queue(&tq, q, g->local_unmap[vert_index]);
      }
    }

    empty_queue(&tq, q);
#pragma omp barrier

#pragma omp single
{   
    clear_recvbuf_vid_data(comm);
    ++level;

    if (debug) printf("Task %d send_size %lu global_size %li\n", 
      procid, temp_send_size, comm->global_queue_size);
}
  } // end while

// resolve conflicts in parents that might occur across tasks
#pragma omp barrier
  for (int32_t i = 0; i < nprocs; ++i)
    tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_sendcounts_thread(g, &tc, i);

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
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_vid_data_queues(g, &tc, comm, i, parents[i]);

  empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
  exchange_vert_data(g, comm, q);
} // end single

#pragma omp for
  for (uint64_t i = 0; i < comm->total_recv; ++i)
  {
    uint64_t vert_index = get_value(g->map, comm->recvbuf_vert[i]);
    parents[vert_index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d bicc_bfs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d bicc_bfs() success\n", procid); }

  return 0;
}



int bicc_bfs_pull(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                   uint64_t* parents, uint64_t* levels, bool* is_leaf, uint64_t root)
{
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
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    is_leaf[i] = true;
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
          if (out_index < g->n_local) {
            parents[vert_index] = g->local_unmap[out_index];
          } else {
            parents[vert_index] = g->ghost_unmap[out_index - g->n_local];
          }
          levels[vert_index] = level;
          is_leaf[out_index] = false;
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
      is_leaf[comm->recvbuf_data[i]] = false;
      levels[vert_index] = level;
    }

#pragma omp single
{   
    clear_recvbuf_vid_data(comm);
    ++level;

    if (debug) printf("Task %d send_size %lu global_size %li\n", 
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

  if (debug) printf("Task %d bicc_bfs_pull() success\n", procid);

  return 0;
}

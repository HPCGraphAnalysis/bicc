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

#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "dist_graph.h"
#include "lca_comms.h"
#include "lca.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;


int bicc_lca(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, 
            uint64_t* parents, uint64_t* levels, 
            uint64_t* highs, uint64_t* high_levels)
{  
  if (debug) { printf("procid %d bicc_lca() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  lca_queue_data_t* lcaq = new lca_queue_data_t;
  init_queue_lca(g, lcaq);

  lcaq->queue_size = 0;
  lcaq->next_size = 0;
  lcaq->finish_size = 0;    
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  uint64_t level = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  lca_thread_data_t lcat;
  thread_comm_t tc;
  init_thread_lca(&lcat);
  init_thread_comm(&tc);

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    if (parents[i] != NULL_KEY)
      highs[i] = parents[i];
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    if (parents[i] != NULL_KEY)
      high_levels[i] = levels[get_value(g->map, parents[i])];

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert_index = i;
    uint64_t vert = g->local_unmap[vert_index];
      
    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      uint64_t out = NULL_KEY;
      if (out_index < g->n_local)
        out = g->local_unmap[out_index];
      else
        out = g->ghost_unmap[out_index - g->n_local];

      if (out > vert) {
        if (highs[out_index] != highs[vert_index] && 
            highs[out_index] != vert && highs[vert_index] != out) {
          add_to_lca(&lcat, lcaq, 
            vert, parents[vert_index], levels[vert_index]-1,
            out, parents[out_index], levels[out_index]-1);
        }
      }
    }
  }

  while (comm->global_queue_size)
  {
    if (debug && lcat.tid == 0) { 
      printf("Task: %d bicc_lca() GQ: %lu, TQ: %lu, FQ: %lu\n", 
        procid, comm->global_queue_size, lcaq->queue_size, lcaq->finish_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->finish_size; i += 3)
    {
      uint64_t vert = lcaq->finish[i];
      uint64_t pred = lcaq->finish[i+1];
      uint64_t level = lcaq->finish[i+2];

      uint64_t vert_index = get_value(g->map, vert);
      highs[vert_index] = pred;
      high_levels[vert_index] = level;
    }
#pragma omp barrier

#pragma omp single
{
    lcaq->finish_size = 0;
}
    
#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->queue_size; i += 6)
    {
      uint64_t vert1 = lcaq->queue[i];
      uint64_t pred1 = lcaq->queue[i+1];
      uint64_t level1 = lcaq->queue[i+2];
      uint64_t vert2 = lcaq->queue[i+3];
      uint64_t pred2 = lcaq->queue[i+4];
      uint64_t level2 = lcaq->queue[i+5];

      uint64_t pred1_index = get_value(g->map, vert1);
      uint64_t pred2_index = get_value(g->map, vert2);
      if (pred1_index != NULL_KEY && pred2_index != NULL_KEY) {
        pred1 = parents[pred1_index];
        pred2 = parents[pred2_index];

        if (pred1 == pred2) {
          add_to_finish(&lcat, lcaq, vert1, pred1, level1-1);
          add_to_finish(&lcat, lcaq, vert2, pred2, level2-1);
        }
        else {
          add_to_lca(&lcat, lcaq, 
            vert1, pred1, level1-1, 
            vert2, pred2, level2-1);
        }
      }
      else if (pred1_index != NULL_KEY) {
        pred1 = parents[pred1_index];

        if (pred1 == pred2) {
          add_to_finish(&lcat, lcaq, vert1, pred1, level1-1);
          add_to_finish(&lcat, lcaq, vert2, pred2, level2-1);
        }
        else {
          add_to_lca(&lcat, lcaq, 
            vert2, pred2, level2, 
            vert1, parents[pred1_index], level1-1);
        }
      }
      else
        printf("Shit fucked\n");
    }  

    empty_lca_queue(&lcat, lcaq);
    empty_finish(&lcat, lcaq);
#pragma omp barrier


// Do the exchange of LCA queue

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->next_size; i+=6)
    {
      uint64_t vert_index = get_value(g->map, lcaq->queue_next[i]);
      if (vert_index < g->n_local)
        tc.sendcounts_thread[procid] += 6;
      else
        tc.sendcounts_thread[g->ghost_tasks[vert_index-g->n_local]] += 6;
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
    init_sendbuf_lca(comm);
}

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->next_size; i+=6)
    {
      uint64_t vert_index = get_value(g->map, lcaq->queue_next[i]);
      int32_t send_rank = -1;
      if (vert_index < g->n_local)
        send_rank = procid;
      else
        send_rank = g->ghost_tasks[vert_index-g->n_local];

      tc.sendcounts_thread[send_rank] += 6;
      update_lca_send(&tc, comm, lcaq, i, send_rank);
    }

    empty_lca_send(&tc, comm, lcaq);
#pragma omp barrier

#pragma omp single
{
    exchange_lca(g, comm);
    memcpy(lcaq->queue, comm->recvbuf_vert, comm->total_recv);
    clear_recvbuf_lca(comm);
    comm->global_queue_size = comm->total_recv;
    lcaq->queue_size = comm->total_recv;
    lcaq->next_size = 0;
} // end single


// Do the exchange of the Finish queue

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->finish_size; i+=3) {      
      uint64_t vert_index = get_value(g->map, lcaq->finish[i]);
      update_sendcounts_thread(g, &tc, vert_index);
    }

    for (int32_t i = 0; i < nprocs; ++i)
    {
#pragma omp atomic
      comm->sendcounts_temp[i] += tc.sendcounts_thread[i]*3;

      tc.sendcounts_thread[i] = 0;
    }
#pragma omp barrier

#pragma omp single
{
    init_sendbuf_lca(comm);
}

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->finish_size; i+=3)
    {
      uint64_t vert_index = get_value(g->map, lcaq->finish[i]);
      int32_t send_rank = -1;
      if (vert_index < g->n_local)
        send_rank = procid;
      else
        send_rank = g->ghost_tasks[vert_index-g->n_local];

      tc.sendcounts_thread[send_rank] += 3;
      update_lca_finish(g, &tc, comm, lcaq, i, send_rank);
    }

    empty_lca_finish(&tc, comm, lcaq);
#pragma omp barrier

#pragma omp single
{
    exchange_lca(g, comm);
    memcpy(lcaq->finish, comm->recvbuf_vert, comm->total_recv);
    clear_recvbuf_lca(comm);
    comm->global_queue_size += comm->total_recv;
    lcaq->finish_size = comm->total_recv;
    uint64_t task_queue_size = comm->global_queue_size;
    MPI_Allreduce(&task_queue_size, &comm->global_queue_size, 1, 
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);  
} // end single

#pragma omp single
{   
    if (debug) printf("Task %d queue %lu finish %lu global_size %li\n", 
      procid, lcaq->queue_size, lcaq->finish_size, comm->global_queue_size);
}
  } // end while

  clear_thread_lca(&lcat);
  clear_thread_comm(&tc);
} // end parallel

  clear_queue_lca(lcaq);
  //delete lcaq;
  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d bicc_bfs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d bicc_bfs() success\n", procid); }

  return 0;
}


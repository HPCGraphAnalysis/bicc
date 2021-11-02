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
            uint64_t* parents, uint64_t* levels, uint64_t* art_pt_flags)
{  
  if (debug) { printf("Rank %d bicc_lca() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  bool* traversed_to_parent = new bool[g->n_total];
  lca_queue_data_t* lcaq = new lca_queue_data_t;
  init_queue_lca(g, lcaq);
  uint64_t bridges = 0;
  uint64_t art_pts = 0;

  lcaq->queue_size = 0;
  lcaq->next_size = 0;
  lcaq->finish_size = 0;    
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  lca_thread_data_t lcat;
  thread_comm_t tc;
  init_thread_lca(&lcat);
  init_thread_comm(&tc);

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    art_pt_flags[i] = 0;
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    traversed_to_parent[i] = false;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert_index = i;
    uint64_t vert = g->local_unmap[vert_index];
      
    uint64_t degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    for (uint64_t j = 0; j < degree; ++j)
    {
      uint64_t out_index = outs[j];
      uint64_t out = NULL_KEY;
      if (out_index < g->n_local)
        out = g->local_unmap[out_index];
      else
        out = g->ghost_unmap[out_index - g->n_local];

      if (vert < out) {
        if (parents[out_index] != vert && parents[vert_index] != out) {

          // printf("Q1 %d, %lu %lu | %lu %lu\n", 
          //  procid, vert,levels[vert_index], out, levels[out_index]);
          if (levels[vert_index] >= levels[out_index])
            add_to_lca(&lcat, lcaq, 
              vert, vert, levels[vert_index],
              out, out, levels[out_index]);
          else
            add_to_lca(&lcat, lcaq, 
              out, out, levels[out_index],
              vert, vert, levels[vert_index]);
        }
      }
    }
  }

  while (comm->global_queue_size)
  {
    if (debug && lcat.tid == 0) { 
      printf("Rank: %d bicc_lca() GQ: %lu, LQ: %lu\n", 
        procid, comm->global_queue_size, lcaq->queue_size); 
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

      // printf("Q %d %lu , %lu %lu %lu | %lu %lu %lu\n", 
      //  procid, i, vert1, pred1, level1, vert2, pred2, level2);
      uint64_t pred1_index = get_value(g->map, pred1);
      uint64_t pred2_index = get_value(g->map, pred2);
      if (pred1 == pred2) {
        art_pt_flags[pred1_index] = 1;
      }
      else if (level1 > level2) {
        // printf("L1 %d - %lu %lu %lu %lu | %lu %lu %lu %lu\n", 
        //   procid, vert1, pred1, level1, parents[pred1_index], 
        //   vert2, pred2, level2, pred2_index);
        pred1 = parents[pred1_index];
        traversed_to_parent[pred1_index] = true;
        add_to_lca(&lcat, lcaq, 
          vert2, pred2, level2, 
          vert1, pred1, level1-1);
      }
      else if (pred1_index != NULL_KEY && pred2_index != NULL_KEY &&
                level1 == level2) {
        // printf("L2 %d - %lu %lu %lu %lu | %lu %lu %lu %lu\n", 
        //   procid, vert1, pred1, level1, parents[pred1_index], 
        //   vert2, pred2, level2, parents[pred2_index]);
        pred1 = parents[pred1_index];
        pred2 = parents[pred2_index];
        traversed_to_parent[pred1_index] = true;
        traversed_to_parent[pred2_index] = true;
        add_to_lca(&lcat, lcaq, 
          vert1, pred1, level1-1, 
          vert2, pred2, level2-1);
      } 
      else if (pred1_index != NULL_KEY) {
        // printf("L1a %d - %lu %lu %lu %lu | %lu %lu %lu %lu\n", 
        //   procid, vert1, pred1, level1, parents[pred1_index], 
        //   vert2, pred2, level2, pred2_index);
        assert(level1 >= level2);
        pred1 = parents[pred1_index];
        traversed_to_parent[pred1_index] = true;
        add_to_lca(&lcat, lcaq, 
          vert2, pred2, level2, 
          vert1, pred1, level1-1);
      }
      else
        printf("Shit fucked %d - %lu %lu %lu %lu | %lu %lu %lu %lu\n", 
          procid, vert1, pred1, level1, pred1_index, 
          vert2, pred2, level2, pred2_index);
    }  

    empty_lca_queue(&lcat, lcaq);
#pragma omp barrier


// Do the exchange of LCA queue

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->next_size; i+=6)
    {
      int32_t send_rank = get_rank(g, lcaq->queue_next[i+1]);
      tc.sendcounts_thread[send_rank] += 6;
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
      int32_t send_rank = get_rank(g, lcaq->queue_next[i+1]);
      tc.sendcounts_thread[send_rank] += 6;
      update_lca_send(&tc, comm, lcaq, i, send_rank);
    }

    empty_lca_send(&tc, comm, lcaq);
#pragma omp barrier

#pragma omp single
{
    exchange_lca(g, comm);
    memcpy(lcaq->queue, comm->recvbuf_vert, comm->total_recv*sizeof(uint64_t));
    clear_recvbuf_lca(comm);
    MPI_Allreduce(&comm->total_recv, &comm->global_queue_size, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    //comm->global_queue_size = comm->total_recv;
    lcaq->queue_size = comm->total_recv;
    lcaq->next_size = 0;
} // end single


#pragma omp single
{   
    if (debug) printf("Rank %d queue %lu global_size %li\n", 
      procid, lcaq->queue_size, comm->global_queue_size);
}
  } // end while

#pragma omp for reduction(+:bridges)
  for (uint64_t i = 0; i < g->n_local; ++i) {
    if (!traversed_to_parent[i]) {
      art_pt_flags[i] = 1;
      uint64_t parent_index = get_value(g->map, parents[i]);
      art_pt_flags[parent_index] = 1;
      if (parent_index >= g->n_local)
        add_to_lca_bridge(&lcat, lcaq, parents[i]);
      ++bridges;
    }
  }
  
  empty_lca_queue_bridge(&lcat, lcaq);
#pragma omp barrier

  // exchange final bridge queue

  for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < lcaq->next_size; ++i)
    {
      int32_t send_rank = get_rank(g, lcaq->queue_next[i]);
      tc.sendcounts_thread[send_rank] += 1;
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
    for (uint64_t i = 0; i < lcaq->next_size; ++i)
    {
      int32_t send_rank = get_rank(g, lcaq->queue_next[i]);
      tc.sendcounts_thread[send_rank] += 1;
      update_lca_send_bridge(&tc, comm, lcaq, i, send_rank);
    }

    empty_lca_send_bridge(&tc, comm, lcaq);
#pragma omp barrier

#pragma omp single
{
    exchange_lca(g, comm);
    memcpy(lcaq->queue, comm->recvbuf_vert, comm->total_recv*sizeof(uint64_t));
    lcaq->queue_size = comm->total_recv;
    clear_recvbuf_lca(comm);
} // end single

#pragma omp single
{   
    if (debug) printf("Rank %d queue %lu potential art pts to set\n", 
      procid, lcaq->queue_size);
}

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < lcaq->queue_size; ++i) {
    uint64_t vert = lcaq->queue[i];
    uint64_t vert_index = get_value(g->map, vert);
    assert(vert_index != NULL_KEY);
    art_pt_flags[vert_index] = 1;
  }
  
#pragma omp for reduction(+:art_pts)
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (art_pt_flags[i] != 0)
      ++art_pts;
    
    
  clear_thread_lca(&lcat);
  clear_thread_comm(&tc);
} // end parallel

  printf("%d: %lu bridges, %lu potential art pts\n", procid, bridges, art_pts);
  
  clear_queue_lca(lcaq);
  //delete lcaq;
  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Rank %d bicc_bfs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Rank %d bicc_bfs() success\n", procid); }

  return 0;
}


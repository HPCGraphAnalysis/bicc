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
//                      Kamesh Madduri    (madduri@cse.psu.edu)ghost_tasks
//
// *****************************************************************************
//@HEADER
*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "comms.h"
#include "lca_comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;



/*void init_queue_lca(dist_graph_t* g, lca_queue_data_t* lcaq)
{  
  if (debug) { printf("Task %d init_queue_lca() start\n", procid); }

  uint64_t queue_size = g->n_local + g->n_ghost;
  lcaq->queue = (uint64_t*)malloc(queue_size*sizeof(uint64_t));
  lcaq->queue_next = (uint64_t*)malloc(queue_size*sizeof(uint64_t));
  lcaq->finish = (uint64_t*)malloc(queue_size*sizeof(uint64_t));
 
  uint64_t* queue;
  uint64_t* queue_next;
  uint64_t* finish;
  if (lcaq->queue == NULL || 
    lcaq->queue_next == NULL || lcaq->queue_send == NULL)
    throw_err("init_queue_lca(), unable to allocate resources\n", procid);

  lcaq->queue_size = 0;
  lcaq->next_size = 0;
  lcaq->finish_size = 0;  

  if (debug) { printf("Task %d init_queue_lca() success\n", procid); }
}

void clear_queue_lca(lca_queue_data_t* lcaq)
{
  if (debug) { printf("Task %d clear_queue_lca() start\n", procid); }

  free(q->queue);
  free(q->queue_next);
  free(q->queue_finish);

  if (debug) { printf("Task %d clear_queue_lca() success\n", procid); }
}

void init_thread_lca(lca_data_thread_t* lcat)
{
  //if (debug) { printf("Task %d init_thread_queue() start\n", procid); }

  lcat->tid = omp_get_thread_num();
  lcat->thread_queue = (uint64_t*)malloc(LCA_THREAD_QUEUE_SIZE*sizeof(uint64_t));
  lcat->thread_finish = (uint64_t*)malloc(LCA_THREAD_QUEUE_SIZE*sizeof(uint64_t));
  if (lcat->thread_queue == NULL || lcat->thread_send == NULL)
    throw_err("init_thread_lca_data(), unable to allocate resources\n", procid, lcat->tid);

  lcat->tid = omp_get_thread_num();
  lcat->thread_queue_size = 0;
  lcat->thread_finish_size = 0;

  //if (debug) { printf("Task %d init_thread_queue() success\n", procid); }
}

void clear_thread_lca(lca_data_thread_t* lcat)
{  
  free(tq->thread_queue);
  free(tq->thread_finish);
}

void init_sendbuf_lca(mpi_data_t* comm)
{
  comm->sdispls_temp[0] = 0;
  comm->total_send = comm->sendcounts_temp[0];
  for (int32_t i = 1; i < nprocs; ++i)
  {
    comm->sdispls_temp[i] = comm->sdispls_temp[i-1] + comm->sendcounts_temp[i-1];
    comm->total_send += comm->sendcounts_temp[i];
  }

  if (debug) printf("Task %d total_send %lu\n", procid, comm->total_send);

  comm->sendbuf_vert = (uint64_t*)malloc(comm->total_send*sizeof(uint64_t));
  if (comm->sendbuf_vert == NULL)
    throw_err("init_sendbuf_vid_data(), unable to allocate resources\n", procid);
}

void clear_recvbuf_lca(mpi_data_t* comm)
{
  free(comm->recvbuf_vert);

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts[i] = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;
}*/


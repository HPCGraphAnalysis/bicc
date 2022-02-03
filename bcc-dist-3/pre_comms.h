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

#ifndef _PRE_COMMS_H_
#define _PRE_COMMS_H_

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

#define PRE_THREAD_QUEUE_SIZE 7168

struct pre_thread_data_t {
  int32_t tid;
  uint64_t* thread_queue;
  uint64_t thread_queue_size;
};

struct pre_queue_data_t {
  uint64_t* queue;
  uint64_t* queue_next;

  uint64_t queue_size;
  uint64_t next_size;

  uint64_t queue_length;
};


inline void init_queue_pre(dist_graph_t* g, pre_queue_data_t* preq){
  if (debug) { printf("Task %d init_queue_pre() start\n", procid);}
  
  preq->queue_length = g->m_local*10;//g->n_local + g->n_ghost;
  preq->queue = (uint64_t*)malloc(preq->queue_length*sizeof(uint64_t));
  preq->queue_next = (uint64_t*)malloc(preq->queue_length*sizeof(uint64_t));
  if (preq->queue == NULL || preq->queue_next == NULL)
    throw_err("init_queue_pre(), unable to allocate resources\n",procid);
  
  preq->queue_size = 0;
  preq->next_size = 0;
  if(debug){printf("Task %d init_queue_pre() success\n", procid); }
}

inline void clear_queue_pre(pre_queue_data_t* preq){
  if(debug){ printf("Task %d clear_queue_pre() start\n",procid); }
  
  free(preq->queue);
  free(preq->queue_next);

  if(debug) {printf("Task %d clear_queue_pre() success\n", procid); }
}

inline void init_thread_pre(pre_thread_data_t* pret) {
  if (debug) { printf("Task %d init_thread_queue() start\n", procid);}

  pret->tid = omp_get_thread_num();
  pret->thread_queue = (uint64_t*)malloc(PRE_THREAD_QUEUE_SIZE*sizeof(uint64_t));
  if (pret->thread_queue == NULL)
    throw_err("init_thread_pre(), unable to allocate resources\n", procid, pret->tid);

  pret->tid = omp_get_thread_num();
  pret->thread_queue_size = 0;
  
  if (debug) {printf("Task %d init_thread_queue() success\n", procid); }
}

inline void clear_thread_pre(pre_thread_data_t* pret){
  free(pret->thread_queue);
}

inline void init_sendbuf_pre(mpi_data_t* comm){
  comm->sdispls_temp[0] = 0;
  comm->total_send = 0;
  comm->total_send = comm->sendcounts_temp[0];
  for (int32_t i = 1; i < nprocs; ++i){
    comm->sdispls_temp[i] = comm->sdispls_temp[i-1] + comm->sendcounts_temp[i-1];
    comm->total_send += comm->sendcounts_temp[i];
  }
  
  if (debug) printf("Task %d total_send %lu\n", procid, comm->total_send);

  comm->sendbuf_vert = (uint64_t*)malloc(comm->total_send*sizeof(uint64_t));
  if (comm->sendbuf_vert == NULL)
    throw_err("init_sendbuf_pre(), unable to allocate resources\n", procid);
}

inline void clear_recvbuf_pre(mpi_data_t* comm){
  free(comm->recvbuf_vert);
  
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts[i] = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;
}


inline void add_to_pre(pre_thread_data_t* pret, pre_queue_data_t* preq, 
                    uint64_t vert, uint64_t rank, uint64_t subscript,
                    uint64_t next, uint64_t next_rank, uint64_t next_subscript,
                    uint64_t count);
inline void empty_pre_queue(pre_thread_data_t* pret, pre_queue_data_t* preq);


inline void update_pre_send(
  thread_comm_t* tc, mpi_data_t* comm,
  pre_queue_data_t* preq, uint64_t index);
inline void empty_pre_send( 
  thread_comm_t* tc, mpi_data_t* comm, pre_queue_data_t* preq);

inline void exchange_pre(dist_graph_t* g, mpi_data_t* comm);



inline void add_to_pre(pre_thread_data_t* pret, pre_queue_data_t* preq, 
                    uint64_t vert, uint64_t rank, uint64_t subscript,
                    uint64_t next, uint64_t next_rank, uint64_t next_subscript,
                    uint64_t count)
{
  pret->thread_queue[pret->thread_queue_size++] = vert;
  pret->thread_queue[pret->thread_queue_size++] = rank;
  pret->thread_queue[pret->thread_queue_size++] = subscript;
  pret->thread_queue[pret->thread_queue_size++] = next;
  pret->thread_queue[pret->thread_queue_size++] = next_rank;
  pret->thread_queue[pret->thread_queue_size++] = next_subscript;
  pret->thread_queue[pret->thread_queue_size++] = count;

  if (pret->thread_queue_size+7 >= PRE_THREAD_QUEUE_SIZE)
    empty_pre_queue(pret, preq);
}

inline void empty_pre_queue(pre_thread_data_t* pret, pre_queue_data_t* preq)
{
  uint64_t start_offset;

#pragma omp atomic capture
  start_offset = preq->next_size += pret->thread_queue_size;

  start_offset -= pret->thread_queue_size;
  for (uint64_t i = 0; i < pret->thread_queue_size; ++i)
    preq->queue_next[start_offset + i] = pret->thread_queue[i];
  pret->thread_queue_size = 0;
}



inline void update_pre_send( 
  thread_comm_t* tc, mpi_data_t* comm,
  pre_queue_data_t* preq, uint64_t index)
{
  // if (preq->queue_next[index+4] != 0) 
  //   printf("BAD RANK %lu %lu %lu %lu %lu %lu %lu\n", 
  //     preq->queue_next[index], preq->queue_next[index+1],
  //     preq->queue_next[index+2], preq->queue_next[index+3],
  //     preq->queue_next[index+4], preq->queue_next[index+5],
  //     preq->queue_next[index+6]);
  
  //tc->sendbuf_rank_thread[tc->thread_queue_size/7] = (int32_t)send_rank;
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index+1];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index+2];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index+3];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index+4];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index+5];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = preq->queue_next[index+6];
  //++tc->thread_queue_size;
  //++tc->sendcounts_thread[send_rank];

  if (tc->thread_queue_size+7 >= PRE_THREAD_QUEUE_SIZE)
    empty_pre_send(tc, comm, preq);
}

inline void empty_pre_send(
  thread_comm_t* tc, mpi_data_t* comm, pre_queue_data_t* preq)
{
for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic capture
    tc->thread_starts[i] = comm->sdispls_temp[i] += tc->sendcounts_thread[i];

    tc->thread_starts[i] -= tc->sendcounts_thread[i];
  }

  for (uint64_t i = 0; i < tc->thread_queue_size; i+=7)
  {
    //int32_t cur_rank = tc->sendbuf_rank_thread[i/7];
    int32_t cur_rank = (int32_t)tc->sendbuf_vert_thread[i+4];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]] = 
        tc->sendbuf_vert_thread[i];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]+1] = 
        tc->sendbuf_vert_thread[i+1];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]+2] = 
        tc->sendbuf_vert_thread[i+2];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]+3] = 
        tc->sendbuf_vert_thread[i+3];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]+4] = 
        tc->sendbuf_vert_thread[i+4];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]+5] = 
        tc->sendbuf_vert_thread[i+5];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]+6] = 
        tc->sendbuf_vert_thread[i+6];
    tc->thread_starts[cur_rank] += 7;
  }
  
  for (int32_t i = 0; i < nprocs; ++i)
  {
    tc->thread_starts[i] = 0;
    tc->sendcounts_thread[i] = 0;
  }
  tc->thread_queue_size = 0;
}


inline void exchange_pre(dist_graph_t* g, mpi_data_t* comm)
{
  for (int32_t i = 0; i < nprocs; ++i)
    comm->recvcounts_temp[i] = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sdispls_temp[i] -= comm->sendcounts_temp[i];

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  comm->total_recv = 0;
  for (int i = 0; i < nprocs; ++i)
    comm->total_recv += comm->recvcounts_temp[i];
  
  if (debug) printf("Task %d total_recv %lu\n", procid, comm->total_recv);

  comm->recvbuf_vert = (uint64_t*)malloc(comm->total_recv*sizeof(uint64_t));
  if (comm->recvbuf_vert == NULL)
    throw_err("exchange_pre() unable to allocate recv buffers", procid);

  uint64_t task_queue_size = comm->total_send;
  uint64_t current_global_size = 0;
  MPI_Allreduce(&task_queue_size, &current_global_size, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  uint64_t num_comms = current_global_size / (uint64_t)MAX_SEND_SIZE + 1;
  uint64_t sum_recv = 0;
  uint64_t sum_send = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];
      comm->sendcounts[i] = (int32_t)(send_end - send_begin);
      assert(comm->sendcounts[i] >= 0);
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* buf_v = (uint64_t*)malloc((uint64_t)(cur_send)*sizeof(uint64_t));
    if (buf_v == NULL)
      throw_err("exchange_verts(), unable to allocate comm buffers", procid);

    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];

      for (uint64_t j = send_begin; j < send_end; ++j)
      {
        uint64_t data = comm->sendbuf_vert[comm->sdispls_temp[i]+j];
        buf_v[comm->sdispls_cpy[i]++] = data;
      }
    }

    MPI_Alltoallv(buf_v, comm->sendcounts, 
                  comm->sdispls, MPI_UINT64_T, 
                  comm->recvbuf_vert+sum_recv, comm->recvcounts, 
                  comm->rdispls, MPI_UINT64_T, MPI_COMM_WORLD);
    free(buf_v);
    sum_recv += cur_recv;
    sum_send += cur_send;
  }
  free(comm->sendbuf_vert);

  assert(sum_recv == comm->total_recv);
  assert(sum_send == comm->total_send);
}


#endif


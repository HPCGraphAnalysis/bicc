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

#ifndef _LCA_COMMS_H_
#define _LCA_COMMS_H_

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "comms.h"
#include "bicc_dist.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

#define MAX_SEND_SIZE 2147483648
#define LCA_THREAD_QUEUE_SIZE 6144

struct lca_thread_data_t {
  int32_t tid;
  uint64_t* thread_queue;
  uint64_t* thread_finish;
  uint64_t thread_queue_size;
  uint64_t thread_finish_size;
};

struct lca_queue_data_t {
  uint64_t* queue;
  uint64_t* queue_next;
  uint64_t* finish;

  uint64_t queue_size;
  uint64_t next_size;
  uint64_t finish_size;
};


inline void init_queue_lca(dist_graph_t* g, lca_queue_data_t* lcaq){
  if (debug) { printf("Task %d init_queue_lca() start\n", procid);}
  
  uint64_t queue_size = g->m_local;//g->n_local + g->n_ghost;
  lcaq->queue = (uint64_t*)malloc(20*queue_size*sizeof(uint64_t));
  lcaq->queue_next = (uint64_t*)malloc(20*queue_size*sizeof(uint64_t));
  lcaq->finish = (uint64_t*)malloc(20*queue_size*sizeof(uint64_t));  
  if (lcaq->queue == NULL || lcaq->queue_next == NULL || lcaq->finish == NULL)
    throw_err("init_queue_lca(), unable to allocate resources\n",procid);
  
  lcaq->queue_size = 0;
  lcaq->next_size = 0;
  lcaq->finish_size = 0;
  if(debug){printf("Task %d init_queue_lca() success\n", procid); }
}

inline void clear_queue_lca(lca_queue_data_t* lcaq){
  if(debug){ printf("Task %d clear_queue_lca() start\n",procid); }
  
  free(lcaq->queue);
  free(lcaq->queue_next);
  free(lcaq->finish);

  if(debug) {printf("Task %d clear_queue_lca() success\n", procid); }
}

inline void init_thread_lca(lca_thread_data_t* lcat) {
  if (debug) { printf("Task %d init_thread_queue() start\n", procid);}

  lcat->tid = omp_get_thread_num();
  lcat->thread_queue = (uint64_t*)malloc(LCA_THREAD_QUEUE_SIZE*sizeof(uint64_t));
  lcat->thread_finish = (uint64_t*)malloc(LCA_THREAD_QUEUE_SIZE*sizeof(uint64_t));
  if (lcat->thread_queue == NULL || lcat->thread_finish == NULL)
    throw_err("init_thread_lca(), unable to allocate resources\n", procid, lcat->tid);

  lcat->tid = omp_get_thread_num();
  lcat->thread_queue_size = 0;
  lcat->thread_finish_size = 0;
  
  if (debug) {printf("Task %d init_thread_queue() success\n", procid); }
}

inline void clear_thread_lca(lca_thread_data_t* lcat){
  free(lcat->thread_queue);
  free(lcat->thread_finish);
}

inline void init_sendbuf_lca(mpi_data_t* comm){
  comm->sdispls_temp[0] = 0;
  comm->total_send = comm->sendcounts_temp[0];
  for (int32_t i = 1; i < nprocs; ++i){
    comm->sdispls_temp[i] = comm->sdispls_temp[i-1] + comm->sendcounts_temp[i-1];
    comm->total_send += comm->sendcounts_temp[i];
  }
  
  if (debug) printf("Task %d total_send %lu\n", procid, comm->total_send);

  comm->sendbuf_vert = (uint64_t*)malloc(comm->total_send*sizeof(uint64_t));
  if (comm->sendbuf_vert == NULL)
    throw_err("init_sendbuf_lca(), unable to allocate resources\n", procid);
}

inline void clear_recvbuf_lca(mpi_data_t* comm){
  free(comm->recvbuf_vert);
  
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts[i] = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;
}

inline void add_to_lca(lca_thread_data_t* lcat, lca_queue_data_t* lcaq, 
                       uint64_t vert1, uint64_t pred1, uint64_t level1,
                       uint64_t vert2, uint64_t pred2, uint64_t level2);
inline void empty_lca_queue(lca_thread_data_t* lcat, lca_queue_data_t* lcaq);

inline void add_to_lca_bridge(
  lca_thread_data_t* lcat, lca_queue_data_t* lcaq, uint64_t vert);
inline void empty_lca_queue_bridge(
  lca_thread_data_t* lcat, lca_queue_data_t* lcaq);


// inline void add_to_finish(lca_thread_data_t* lcat, lca_queue_data_t* lcaq, 
//                        uint64_t vert1, uint64_t pred1, uint64_t level1);
// inline void empty_finish_queue(lca_thread_data_t* lcat, lca_queue_data_t* lcaq);

inline void update_lca_send(
  thread_comm_t* tc, mpi_data_t* comm,
  lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank);
inline void empty_lca_send( 
  thread_comm_t* tc, mpi_data_t* comm, lca_queue_data_t* lcaq);

inline void update_lca_send_bridge(
  thread_comm_t* tc, mpi_data_t* comm,
  lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank);
inline void empty_lca_send_bridge( 
  thread_comm_t* tc, mpi_data_t* comm, lca_queue_data_t* lcaq);

// inline void update_lca_finish(dist_graph_t* g, 
//   thread_comm_t* tc, mpi_data_t* comm,
//   lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank);
//(dist_graph_t* g, thread_comm_t* tc, mpi_data_t* comm,
//  lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank);
// inline void empty_lca_finish(dist_graph_t* g, 
//   thread_comm_t* tc, mpi_data_t* comm, lca_queue_data_t* lcaq);

inline void exchange_lca(dist_graph_t* g, mpi_data_t* comm);



inline void add_to_lca(lca_thread_data_t* lcat, lca_queue_data_t* lcaq, 
                       uint64_t vert1, uint64_t pred1, uint64_t level1,
                       uint64_t vert2, uint64_t pred2, uint64_t level2)
{
  lcat->thread_queue[lcat->thread_queue_size++] = vert1;
  lcat->thread_queue[lcat->thread_queue_size++] = pred1;
  lcat->thread_queue[lcat->thread_queue_size++] = level1;
  lcat->thread_queue[lcat->thread_queue_size++] = vert2;
  lcat->thread_queue[lcat->thread_queue_size++] = pred2;
  lcat->thread_queue[lcat->thread_queue_size++] = level2;

  if (lcat->thread_queue_size+6 >= LCA_THREAD_QUEUE_SIZE)
    empty_lca_queue(lcat, lcaq);
}

inline void empty_lca_queue(lca_thread_data_t* lcat, lca_queue_data_t* lcaq)
{
  uint64_t start_offset;

#pragma omp atomic capture
  start_offset = lcaq->next_size += lcat->thread_queue_size;

  start_offset -= lcat->thread_queue_size;
  for (uint64_t i = 0; i < lcat->thread_queue_size; ++i)
    lcaq->queue_next[start_offset + i] = lcat->thread_queue[i];
  lcat->thread_queue_size = 0;
}


inline void add_to_lca_bridge(
  lca_thread_data_t* lcat, lca_queue_data_t* lcaq, uint64_t vert)
{
  lcat->thread_queue[lcat->thread_queue_size++] = vert;

  if (lcat->thread_queue_size+1 >= LCA_THREAD_QUEUE_SIZE)
    empty_lca_queue_bridge(lcat, lcaq);
}

inline void empty_lca_queue_bridge(
  lca_thread_data_t* lcat, lca_queue_data_t* lcaq)
{
  uint64_t start_offset;

#pragma omp atomic capture
  start_offset = lcaq->next_size += lcat->thread_queue_size;

  start_offset -= lcat->thread_queue_size;
  for (uint64_t i = 0; i < lcat->thread_queue_size; ++i)
    lcaq->queue_next[start_offset + i] = lcat->thread_queue[i];
  lcat->thread_queue_size = 0;
}


// inline void add_to_finish(lca_thread_data_t* lcat, lca_queue_data_t* lcaq, 
//                        uint64_t vert1, uint64_t pred1, uint64_t level1)
// {
//   lcat->thread_finish[lcat->thread_finish_size++] = vert1;
//   lcat->thread_finish[lcat->thread_finish_size++] = pred1;
//   lcat->thread_finish[lcat->thread_finish_size++] = level1;

//   if (lcat->thread_finish_size+3 >= LCA_THREAD_QUEUE_SIZE)
//     empty_finish_queue(lcat, lcaq);
// }

// inline void empty_finish_queue(lca_thread_data_t* lcat, lca_queue_data_t* lcaq)
// {
//   uint64_t start_offset;

// #pragma omp atomic capture
//   start_offset = lcaq->finish_size += lcat->thread_finish_size;

//   start_offset -= lcat->thread_finish_size;
//   for (uint64_t i = 0; i < lcat->thread_finish_size; ++i)
//     lcaq->finish[start_offset + i] = lcat->thread_finish[i];
//   lcat->thread_finish_size = 0;
// }

inline void update_lca_send( 
  thread_comm_t* tc, mpi_data_t* comm,
  lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank)
{
  tc->sendbuf_rank_thread[tc->thread_queue_size/6] = send_rank;
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+1];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+2];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+3];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+4];
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+5];
  //++tc->thread_queue_size;
  //++tc->sendcounts_thread[send_rank];

  if (tc->thread_queue_size+6 >= LCA_THREAD_QUEUE_SIZE)
    empty_lca_send(tc, comm, lcaq);
}

inline void empty_lca_send(
  thread_comm_t* tc, mpi_data_t* comm, lca_queue_data_t* lcaq)
{
for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic capture
    tc->thread_starts[i] = comm->sdispls_temp[i] += tc->sendcounts_thread[i];

    tc->thread_starts[i] -= tc->sendcounts_thread[i];
  }

  for (uint64_t i = 0; i < tc->thread_queue_size; i+=6)
  {
    int32_t cur_rank = tc->sendbuf_rank_thread[i/6];
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
    tc->thread_starts[cur_rank] += 6;
  }
  
  for (int32_t i = 0; i < nprocs; ++i)
  {
    tc->thread_starts[i] = 0;
    tc->sendcounts_thread[i] = 0;
  }
  tc->thread_queue_size = 0;
}

inline void update_lca_send_bridge( 
  thread_comm_t* tc, mpi_data_t* comm,
  lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank)
{
  tc->sendbuf_rank_thread[tc->thread_queue_size] = send_rank;
  tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index];

  if (tc->thread_queue_size+1 >= LCA_THREAD_QUEUE_SIZE)
    empty_lca_send_bridge(tc, comm, lcaq);
}

inline void empty_lca_send_bridge(
  thread_comm_t* tc, mpi_data_t* comm, lca_queue_data_t* lcaq)
{
for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic capture
    tc->thread_starts[i] = comm->sdispls_temp[i] += tc->sendcounts_thread[i];

    tc->thread_starts[i] -= tc->sendcounts_thread[i];
  }

  for (uint64_t i = 0; i < tc->thread_queue_size; ++i)
  {
    int32_t cur_rank = tc->sendbuf_rank_thread[i];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]] = 
      tc->sendbuf_vert_thread[i];
    tc->thread_starts[cur_rank] += 1;
  }
  
  for (int32_t i = 0; i < nprocs; ++i)
  {
    tc->thread_starts[i] = 0;
    tc->sendcounts_thread[i] = 0;
  }
  tc->thread_queue_size = 0;
}
// inline void update_lca_finish(dist_graph_t* g, 
//   thread_comm_t* tc, mpi_data_t* comm,
//   lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank)
// {
//   // for (int32_t i = 0; i < nprocs; ++i)
//   //   tc->v_to_rank[i] = false;

//   // uint64_t out_degree = out_degree(g, vert_index);
//   // uint64_t* outs = out_vertices(g, vert_index);
//   // for (uint64_t j = 0; j < out_degree; ++j)
//   // {
//   //   uint64_t out_index = outs[j];
//   //   if (out_index >= g->n_local)
//   //   {
//   //     int32_t out_rank = g->ghost_tasks[out_index - g->n_local];
//   //     if (!tc->v_to_rank[out_rank])
//   //     {
//   //       tc->v_to_rank[out_rank] = true;
//   //       add_vid_data_to_send(tc, comm,
//   //         g->local_unmap[vert_index], data, out_rank);
//   //     }
//   //   }
//   // }  
//   //tc->sendbuf_rank_thread[tc->thread_queue_size/3] = send_rank;
//   tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->finish[index];
//   tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->finish[index+1];
//   tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->finish[index+2];
//   //++tc->thread_queue_size;
//   //++tc->sendcounts_thread[send_rank];

//   if (tc->thread_queue_size+6 >= LCA_THREAD_QUEUE_SIZE)
//     empty_lca_finish(g, tc, comm, lcaq);
// }

// inline void add_data_to_finish(thread_comm_t* tc, mpi_data_t* comm,
//   lca_queue_data_t* lcaq, uint64_t index, int32_t send_rank)
// {
//   tc->sendbuf_rank_thread[tc->thread_queue_size/3] = send_rank;
//   tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index];
//   tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+1];
//   tc->sendbuf_vert_thread[tc->thread_queue_size++] = lcaq->queue_next[index+2];
//   ++tc->thread_queue_size;
//   ++tc->sendcounts_thread[send_rank];

//   if (tc->thread_queue_size+3 >= LCA_THREAD_QUEUE_SIZE)
//     empty_lca_finish(tc, comm, lcaq);
// }

// inline void empty_lca_finish(dist_graph_t* g, 
//   thread_comm_t* tc, mpi_data_t* comm, lca_queue_data_t* lcaq)
// {
// for (int32_t i = 0; i < nprocs; ++i)
//   {
// #pragma omp atomic capture
//     tc->thread_starts[i] = comm->sdispls_temp[i] += tc->sendcounts_thread[i];

//     tc->thread_starts[i] -= tc->sendcounts_thread[i];
//   }

//   for (uint64_t i = 0; i < tc->thread_queue_size; i+=3)
//   {
//     int32_t cur_rank = get_rank(g, tc->sendbuf_vert_thread[i]);
//     comm->sendbuf_vert[tc->thread_starts[cur_rank]] = 
//       tc->sendbuf_vert_thread[i];
//     comm->sendbuf_vert[tc->thread_starts[cur_rank]+1] = 
//       tc->sendbuf_vert_thread[i+1];
//     comm->sendbuf_vert[tc->thread_starts[cur_rank]+2] = 
//       tc->sendbuf_vert_thread[i+2];
//     tc->thread_starts[cur_rank] += 3;
//   }
  
//   for (int32_t i = 0; i < nprocs; ++i)
//   {
//     tc->thread_starts[i] = 0;
//     tc->sendcounts_thread[i] = 0;
//   }
//   tc->thread_queue_size = 0;
// }

inline void exchange_lca(dist_graph_t* g, mpi_data_t* comm)
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
    throw_err("exchange_lca() unable to allocate recv buffers", procid);

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


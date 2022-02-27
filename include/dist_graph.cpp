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
#include <string.h>

#include "fast_map.h"
#include "dist_graph.h"
#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

int create_graph(graph_gen_data_t *ggi, dist_graph_t *g)
{  
  if (debug) { printf("Task %d create_graph() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = ggi->n_offset;
  g->m = ggi->m;
  g->m_local = ggi->m_local_edges;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));
  g->n_offsets = (uint64_t*)malloc((nprocs+1)*sizeof(uint64_t));
  MPI_Allgather(&g->n_offset, 1, MPI_UINT64_T, g->n_offsets, 1, 
    MPI_UINT64_T, MPI_COMM_WORLD);
  g->n_offsets[nprocs] = g->n;

  if(ggi->global_edge_indices != NULL) {
    g->edge_map = (struct fast_map*)malloc(sizeof(struct fast_map));
    g->edge_unmap = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  }

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  uint64_t* out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_degree_list == NULL || temp_counts == NULL)
    throw_err("create_graph(), unable to allocate graph edge storage", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_degree_list[i] = 0;
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local*2; i+=2)
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, g->n_local*sizeof(uint64_t));


  for (uint64_t i = 0; i < g->m_local*2; i+=2){
    out_edges[temp_counts[ggi->gen_edges[i] - g->n_offset]] = ggi->gen_edges[i+1];
    if(ggi->global_edge_indices != NULL){
      g->edge_unmap[temp_counts[ggi->gen_edges[i] - g->n_offset]++] = ggi->global_edge_indices[i/2];
    } else {
      temp_counts[ggi->gen_edges[i] - g->n_offset]++;
    }
  }

  if(ggi->global_edge_indices != NULL){
    init_map(g->edge_map, g->m_local*2);
    for(uint64_t i = 0; i < g->m_local; i++){
      if(get_value(g->edge_map, g->edge_unmap[i]) == NULL_KEY){
        set_value(g->edge_map, g->edge_unmap[i], i);
      }
    }
  }
  free(ggi->gen_edges);
  free(ggi->global_edge_indices);
  free(temp_counts);
  g->out_edges = out_edges;
  g->out_degree_list = out_degree_list;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (g->local_unmap == NULL)
    throw_err("create_graph(), unable to allocate unmap", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d create_graph() success\n", procid); }
  return 0;
}

int create_graph_serial(graph_gen_data_t *ggi, dist_graph_t *g)
{
  if (debug) { printf("Task %d create_graph_serial() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = 0;
  g->m = ggi->m;
  g->m_local = ggi->m_local_read*2;
  g->n_ghost = 0;
  g->n_total = g->n_local;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));
  g->n_offsets = (uint64_t*)malloc(2*sizeof(uint64_t));
  g->n_offsets[0] = 0;
  g->n_offsets[1] = g->n_local;


  uint64_t* out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  uint64_t* out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));

  if (ggi->global_edge_indices != NULL){
    g->edge_map = (struct fast_map*)malloc(sizeof(struct fast_map));
    g->edge_unmap = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  }
      
  if (out_edges == NULL || out_degree_list == NULL || temp_counts == NULL)
  throw_err("create_graph_serial(), unable to allocate out edge storage\n", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_degree_list[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < ggi->m_local_read*2; i+=2) {
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
    ++temp_counts[ggi->gen_edges[i+1] - g->n_offset];
  }
  for (uint64_t i = 0; i < g->n_local; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, g->n_local*sizeof(uint64_t));
  for (uint64_t i = 0; i < ggi->m_local_read*2; i+=2) {
    if(ggi->global_edge_indices != NULL){
      g->edge_unmap[temp_counts[ggi->gen_edges[i]-g->n_offset]] = ggi->global_edge_indices[i/2];
    }
    out_edges[temp_counts[ggi->gen_edges[i] - g->n_offset]++] = ggi->gen_edges[i+1];
    if(ggi->global_edge_indices != NULL){
      g->edge_unmap[temp_counts[ggi->gen_edges[i+1]-g->n_offset]] = ggi->global_edge_indices[i/2];
    }
    out_edges[temp_counts[ggi->gen_edges[i+1] - g->n_offset]++] = ggi->gen_edges[i];
  }

  if(ggi->global_edge_indices != NULL){
    init_map_nohash(g->edge_map, g->m_local*2);
    for(uint64_t i = 0; i < g->m_local; i+=2){
      if(get_value(g->edge_map, g->edge_unmap[i/2]) == NULL_KEY){
        set_value(g->edge_map, g->edge_unmap[i/2],i/2);
      }
    }
  }

  free(ggi->gen_edges);
  free(ggi->global_edge_indices);
  free(temp_counts);
  g->out_edges = out_edges;
  g->out_degree_list = out_degree_list;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));  
  if (g->local_unmap == NULL)
    throw_err("create_graph_serial(), unable to allocate unmap\n", procid);

  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  //int64_t total_edges = g->m_local_in + g->m_local_out;
  init_map_nohash(g->map, g->n);
  for (uint64_t i = 0; i < g->n_local; ++i)
    set_value_uq(g->map, i, i);


  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph_serial() %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d create_graph_serial() success\n", procid); }
  return 0;
}

int create_graph(dist_graph_t* g, 
          uint64_t n_global, uint64_t m_global, 
          uint64_t n_local, uint64_t m_local,
          uint64_t* local_offsets, uint64_t* local_adjs, 
          uint64_t* global_ids)
{ 
  if (debug) { printf("Task %d create_graph() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = n_global;
  g->n_local = n_local;
  g->m = m_global;
  g->m_local = m_local;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));

  g->out_edges = local_adjs;
  g->out_degree_list = local_offsets;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (g->local_unmap == NULL)
    throw_err("create_graph(), unable to allocate unmap", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = global_ids[i];

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d create_graph() success\n", procid); }
  return 0;
}


int create_graph_serial(dist_graph_t* g, 
          uint64_t n_global, uint64_t m_global, 
          uint64_t n_local, uint64_t m_local,
          uint64_t* local_offsets, uint64_t* local_adjs)
{
  if (debug) { printf("Task %d create_graph_serial() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = n_global; printf("N global %lu %lu\n", n_global, g->n);
  g->n_local = n_local;
  g->n_offset = 0;
  g->n_ghost = 0;
  g->m = m_global;
  g->m_local = m_local;
  g->n_total = g->n_local;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));

  g->out_edges = local_adjs;
  g->out_degree_list = local_offsets;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));  
  if (g->local_unmap == NULL)
    throw_err("create_graph_serial(), unable to allocate unmap\n", procid);

  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  init_map_nohash(g->map, g->n);
  for (uint64_t i = 0; i < g->n_local; ++i)
    set_value_uq(g->map, i, i);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph_serial() %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d create_graph_serial() success\n", procid); }
  return 0;
}


int clear_graph(dist_graph_t *g)
{
  if (debug) { printf("Task %d clear_graph() start\n", procid); }

  free(g->out_edges);
  free(g->out_degree_list);
  //free(g->ghost_degrees);
  free(g->local_unmap);
  if (nprocs > 1) {
    free(g->ghost_unmap);
    free(g->ghost_tasks);
  }
  clear_map(g->map);
  free(g->map);

  if (debug) { printf("Task %d clear_graph() success\n", procid); }
  return 0;
} 


int relabel_edges(dist_graph_t *g)
{
  relabel_edges(g, NULL);
  return 0;
}


int relabel_edges(dist_graph_t *g, int32_t* part_list)
{
  if (debug) { printf("Task %d relabel_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t cur_label = g->n_local;
  uint64_t total_edges = g->m_local + g->n_local;

  if (total_edges*2 < g->n)
    init_map_nohash(g->map, g->n);
  else
    init_map(g->map, total_edges*2);

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert = g->local_unmap[i];
    set_value(g->map, vert, i);
  }

  for (uint64_t i = 0; i < g->m_local; ++i)
  {
    uint64_t out = g->out_edges[i];
    uint64_t val = get_value(g->map, out);
    if (val == NULL_KEY)
    {
      set_value_uq(g->map, out, cur_label);
      g->out_edges[i] = cur_label++;
    }
    else        
      g->out_edges[i] = val;
  }

  g->n_ghost = g->map->num_unique;
  g->n_total = g->n_ghost + g->n_local;

  if (debug)
    printf("Task %d, n_ghost %lu\n", procid, g->n_ghost);

  g->ghost_unmap = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  g->ghost_tasks = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  if (g->ghost_unmap == NULL || g->ghost_tasks == NULL)
    throw_err("relabel_edges(), unable to allocate ghost unmaps", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
  {
    uint64_t cur_index = get_value(g->map, g->map->unique_keys[i]);

    cur_index -= g->n_local;
    g->ghost_unmap[cur_index] = g->map->unique_keys[i];
  }

  if (part_list == NULL)
  {
    uint64_t n_per_rank = g->n / (uint64_t)nprocs + 1;  

#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_ghost; ++i)
      g->ghost_tasks[i] = g->ghost_unmap[i] / n_per_rank;
  }
  else
  {
 #pragma omp parallel for
    for (uint64_t i = 0; i < g->n_ghost; ++i)
    {   
      uint64_t global_id = g->ghost_unmap[i];
      int32_t rank = part_list[global_id];
      g->ghost_tasks[i] = rank;
    }
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf(" Task %d relabel_edges() %9.6f (s)\n", procid, elt); 
  }

  if (debug) { printf("Task %d relabel_edges() success\n", procid); }
  return 0;
}



int get_max_degree_vert(dist_graph_t *g)
{ 
  if (debug) { printf("Task %d get_max_degree_vert() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t my_max_degree = 0;
  uint64_t my_max_vert = -1;
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t this_degree = out_degree(g, i);
    if (this_degree > my_max_degree)
    {
      my_max_degree = this_degree;
      my_max_vert = g->local_unmap[i];
    }
  }

  uint64_t max_degree;
  uint64_t max_vert;

  MPI_Allreduce(&my_max_degree, &max_degree, 1, MPI_UINT64_T,
                MPI_MAX, MPI_COMM_WORLD);
  if (my_max_degree == max_degree)
    max_vert = my_max_vert;
  else
    max_vert = NULL_KEY;
  MPI_Allreduce(MPI_IN_PLACE, &max_vert, 1, MPI_UINT64_T,
                MPI_MIN, MPI_COMM_WORLD);

  g->max_degree_vert = max_vert;
  g->max_degree = max_degree;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, max_degree %lu, max_vert %lu, %f (s)\n", 
           procid, max_degree, max_vert, elt);
  }

  if (debug) { printf("Task %d get_max_degree_vert() success\n", procid); }
  return 0;
}


int get_ghost_degrees(dist_graph_t* g)
{
  mpi_data_t comm;
  queue_data_t q;
  init_comm_data(&comm);
  init_queue_data(g, &q);

  get_ghost_degrees(g, &comm, &q);

  clear_comm_data(&comm);
  clear_queue_data(&q);

  return 0;
}


int get_ghost_degrees(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q)
{
  if (debug) { printf("Task %d get_ghost_degrees() start\n", procid); }

  g->ghost_degrees = (uint64_t*)malloc(g->n_ghost*(sizeof(uint64_t)));
  if (g->ghost_degrees == NULL)
    throw_err("get_ghost_degrees(), unable to allocate ghost degrees\n", procid);

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

#pragma omp parallel 
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

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
    update_vid_data_queues(g, &tc, comm, i, 
                           (out_degree(g, i)));

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
    assert(index >= g->n_local);
    assert(index < g->n_total);
    g->ghost_degrees[index - g->n_local] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel


  if (debug) { printf("Task %d get_ghost_degrees() success\n", procid); }

  return 0;
}


int repart_graph(dist_graph_t *g, mpi_data_t* comm, int32_t* part_list)
{
   for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = part_list[g->local_unmap[i]];
    ++comm->sendcounts_temp[rank];
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_vids = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  uint64_t* recvbuf_deg = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_vids == NULL || recvbuf_deg == NULL)
    throw_err("repart_graph(), unable to allocate buffers\n", procid);

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE/(g->m/g->n))+ 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);

  uint64_t sum_recv_deg = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    if (debug)
      printf("Task %d send_begin %lu send_end %lu\n", procid, send_begin, send_end);
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = part_list[g->local_unmap[i]];
      ++comm->sendcounts[rank];
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
    uint64_t* sendbuf_vids = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    uint64_t* sendbuf_deg = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_vids == NULL || sendbuf_deg == NULL)
      throw_err("repart_graph(), unable to allocate buffers\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = part_list[g->local_unmap[i]];
      int32_t snd_index = comm->sdispls_cpy[rank]++;
      sendbuf_vids[snd_index] = g->local_unmap[i];
      sendbuf_deg[snd_index] = (uint64_t)out_degree(g, i);
    }

    MPI_Alltoallv(
      sendbuf_vids, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_vids+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);    
    MPI_Alltoallv(
      sendbuf_deg, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_deg+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_deg += (uint64_t)cur_recv;
    free(sendbuf_vids);
    free(sendbuf_deg);
  }

  for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = part_list[g->local_unmap[i]];
    comm->sendcounts_temp[rank] += (uint64_t)out_degree(g, i);
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  total_recv = 0;
  total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_e_out = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_e_out == NULL)
    throw_err("repart_graph(), unable to allocate buffer\n", procid);

  // max_transfer = total_send > total_recv ? total_send : total_recv;
  // num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  // MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
  //               MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);
  
  uint64_t sum_recv_e_out = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint32_t rank = part_list[g->local_unmap[i]];
      comm->sendcounts[rank] += (int32_t)out_degree(g, i);
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
    uint64_t* sendbuf_e_out = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_e_out == NULL)
      throw_err("repart_graph(), unable to allocate buffer\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t out_degree = out_degree(g, i);
      uint64_t* outs = out_vertices(g, i);
      int32_t rank = part_list[g->local_unmap[i]];
      int32_t snd_index = comm->sdispls_cpy[rank];
      comm->sdispls_cpy[rank] += out_degree;
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out;
        if (outs[j] < g->n_local)
          out = g->local_unmap[outs[j]];
        else
          out = g->ghost_unmap[outs[j]-g->n_local];
        sendbuf_e_out[snd_index++] = out;
      }
    }

    MPI_Alltoallv(sendbuf_e_out, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf_e_out+sum_recv_e_out, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_e_out += (uint64_t)cur_recv;
    free(sendbuf_e_out);
  }

  free(g->out_edges);
  free(g->out_degree_list);
  g->out_edges = recvbuf_e_out;
  g->m_local = (uint64_t)sum_recv_e_out;
  g->out_degree_list = (uint64_t*)malloc((sum_recv_deg+1)*sizeof(uint64_t));
  g->out_degree_list[0] = 0;
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->out_degree_list[i+1] = g->out_degree_list[i] + recvbuf_deg[i];
  assert(g->out_degree_list[sum_recv_deg] == g->m_local);
  free(recvbuf_deg);


  free(g->local_unmap);
  g->local_unmap = (uint64_t*)malloc(sum_recv_deg*sizeof(uint64_t));
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->local_unmap[i] = recvbuf_vids[i];
  free(recvbuf_vids);

  g->n_local = sum_recv_deg;
  MPI_Exscan(&g->n_local, &g->n_offset, 1, MPI_UINT64_T,
              MPI_SUM, MPI_COMM_WORLD);
  if (procid == 0) g->n_offset = 0;
  
  printf("%d - n_local %lu, m_local %lu\n", procid, g->n_local, g->m_local);
  
  free(g->ghost_unmap);
  free(g->ghost_tasks);
  clear_map(g->map);
  relabel_edges(g, part_list);
  delete [] part_list;
      
  return 0;
}


int determine_edge_block(dist_graph_t* g, int32_t*& part_list)
{
  uint64_t* global_degrees = new uint64_t[g->n];  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_degrees[i] = 0;
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_degrees[g->local_unmap[i]] = out_degree(g, i);
  
  MPI_Allreduce(MPI_IN_PLACE, global_degrees, g->n, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  part_list = new int32_t[g->n];
  uint64_t m_per_rank = g->m*2 / (uint64_t)nprocs;
  uint64_t running_sum = 0;
  int32_t cur_rank = 0;
  g->n_offsets[0] = 0;
  for (uint64_t i = 0; i < g->n; ++i) {
    part_list[i] = cur_rank;
    
    running_sum += global_degrees[i];
    if (running_sum > m_per_rank) {
      running_sum = 0;
      cur_rank++;
      g->n_offsets[cur_rank] = i+1;
    }
  }
  g->n_offsets[nprocs] = g->n;
  
  delete [] global_degrees;
  
  // for (int i = 0; i < g->n_local; ++i)
  //   printf("rank %d, vid %lu, rank %d\n", 
  //     procid, g->local_unmap[i], part_list[i]);
   
  return 0;   
}


#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "dist_graph.h"
#include "io_pp.h"
#include "comms.h"
#include "util.h"
#include "reduce_graph.h"
#include "thread.h"
#include "pre_comms.h"
#include "min_max_size.h"

int get_min_max_size(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
  uint64_t* parents, uint64_t* preorders, 
  uint64_t* num_descendents, uint64_t* mins, uint64_t* maxes)
{
  if (debug) { printf("procid %d get_min_max_size() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }
  
  // queue for passing min/max
  pre_queue_data_t* preq = new pre_queue_data_t;
  init_queue_pre(g, preq);
  preq->queue_size = 0;
  preq->next_size = 0;
  
  
  comm->global_queue_size = 1;
#pragma omp parallel
{
  pre_thread_data_t pret;
  thread_comm_t tc;
  init_thread_pre(&pret);
  init_thread_comm(&tc);

  // everyone initialize their min/max and size
#pragma omp for
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    num_descendents[vert_index] = 1;
#pragma omp for
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    mins[vert_index] = preorders[vert_index];
#pragma omp for
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    mins[vert_index] = preorders[vert_index];
  
#pragma omp for schedule(guided)
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    uint64_t vert_preorder = preorders[vert_index];
    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    for (uint64_t j = 0; j < out_degree; ++j) {
      uint64_t out_preorder = preorders[outs[j]];
      if (out_preorder > vert_preorder)
        maxes[vert_index] = out_preorder;
      if (out_preorder < vert_preorder)
        mins[vert_index] = out_preorder;
    }
  }
  
  // leaves add their stuff to the queue
  // note: can reduce queue package size to make more efficient
  // {min, max, count, parent, parent_rank, 0, 0}
  
#pragma omp for schedule(guided)
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    uint64_t vert = g->local_unmap[vert_index];
    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    uint64_t num_children = 0;
    for (uint64_t j = 0; j < out_degree; ++j) {
      if (parents[outs[j]] == vert)
        ++num_children;
    }
    
    if (num_children == 0) {
      uint64_t parent = parents[vert_index];
      uint64_t parent_index = get_value(g->map, parent);
      uint64_t parent_rank = NULL_KEY;
      if (parent_index < g->n_local) 
        parent_rank = (uint64_t)procid;
      else
        parent_rank = g->ghost_tasks[parent_index - g->n_local];
      
      add_to_pre(&pret, preq,
        mins[vert_index], maxes[vert_index], 1, 
        parent, parent_rank, 0, 0);
    }
  }
  
  while (comm->global_queue_size > 0) {
    
    // do communication
       
    // exchange passing data
    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < preq->next_size; i+=7)
    {
      int32_t send_rank = (int32_t)preq->queue_next[i+4];
      tc.sendcounts_thread[send_rank] += 7;
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
    init_sendbuf_pre(comm);
}

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < preq->next_size; i+=7)
    {
      int32_t send_rank = (int32_t)preq->queue_next[i+4];
      tc.sendcounts_thread[send_rank] += 7;
      update_pre_send(&tc, comm, preq, i);
    }

    empty_pre_send(&tc, comm, preq);
#pragma omp barrier

#pragma omp single
{
    exchange_pre(g, comm);
    if (comm->total_recv >= preq->queue_length) {
      if (debug) printf("%d realloc: %li %li\n", procid, comm->total_recv, preq->queue_length);
      preq->queue_length = comm->total_recv;
      preq->queue = (uint64_t*)realloc(preq->queue, preq->queue_length*sizeof(uint64_t));
      preq->queue_next = (uint64_t*)realloc(preq->queue_next, preq->queue_length*sizeof(uint64_t));
      if (preq->queue == NULL || preq->queue_next == NULL)
        throw_err("realloc() queues, unable to allocate resources\n",procid);
    }
    memcpy(preq->queue, comm->recvbuf_vert, comm->total_recv*sizeof(uint64_t));
    clear_recvbuf_pre(comm);
    preq->queue_size = comm->total_recv;
    preq->next_size = 0;
} // end single
   
    if (debug && pret.tid == 0) { 
      printf("Rank: %d get_min_max_size() GQ: %lu, LQ: %lu\n", 
        procid, comm->global_queue_size, preq->queue_size); 
    }
  
    // update values and pass on
#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < preq->queue_size; i += 7) {
      uint64_t min = preq->queue[i];
      uint64_t max = preq->queue[i+1];
      uint64_t count = preq->queue[i+2];
      uint64_t vert = preq->queue[i+3];
      uint64_t vert_index = get_value(g->map, vert);
      
      // we only add ourselves to count if we haven't passed a package
      // otherwise we'd be double counting all the way up the tree
      if (num_descendents[vert_index] == 1)
        count += 1;
      
      num_descendents[vert_index] += count;
      if (max > maxes[vert_index])
        maxes[vert_index] = max;
      if (min < mins[vert_index])
        mins[vert_index] = min;
      
      if (parents[vert_index] != vert) {
        uint64_t parent = parents[vert_index];
        uint64_t parent_index = get_value(g->map, parent);
        uint64_t parent_rank = NULL_KEY;
        if (parent_index < g->n_local) 
          parent_rank = (uint64_t)procid;
        else
          parent_rank = g->ghost_tasks[parent_index - g->n_local];
        
        add_to_pre(&pret, preq,
          mins[vert_index], maxes[vert_index], count, 
          parent, parent_rank, 0, 0);
      }
    }

    empty_pre_queue(&pret, preq);
#pragma omp barrier
    
    // keep track of how many in queue
    // need to do this since we had to do comms first in while loop
#pragma omp single
{   
    comm->global_queue_size = 0;
    MPI_Allreduce(&preq->next_size, &comm->global_queue_size, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
}    
  } // end while

  //clear_thread_lca(&lcat);
  clear_thread_comm(&tc);

} // end parallel

  //clear_queue_lca(lcaq);
  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Rank %d get_min_max_size() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Rank %d get_min_max_size() success\n", procid); }

  return 0;
}


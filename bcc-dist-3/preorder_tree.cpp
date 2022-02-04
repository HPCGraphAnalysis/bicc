
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

int preorder_tree(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
  uint64_t* parents, uint64_t* levels, 
  uint64_t* preorders, uint64_t* num_descendents)
{
  if (debug) { printf("procid %d preorder_tree() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }
  
  // queue for exchanging parent information
  q = (queue_data_t*)malloc(sizeof(queue_data_t));;
  init_queue_data(g, q);

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  // queue for exchanging pointer jumping information
  pre_queue_data_t* preq = new pre_queue_data_t;
  init_queue_pre(g, preq);

  preq->queue_size = 0;
  preq->next_size = 0;
  
  // num children of each vertex
  uint64_t* num_children = new uint64_t[g->n_local];
  
  // subscript of each vertex relative to their parent
  uint64_t* subscripts = new uint64_t[g->n_local];
  
  // 2d next, ranks, counts array
  uint64_t** nexts = new uint64_t*[g->n_local];
  uint64_t** next_subs = new uint64_t*[g->n_local];
  uint64_t** counts = new uint64_t*[g->n_local];
  uint64_t** ranks = new uint64_t*[g->n_local];
  bool* visited = new bool[g->n_total];
  
  // pre-order specific computations
  uint64_t max_preorder = 0;
  uint64_t min_preorder = 0;
  uint64_t num_per_rank = 0;
  uint64_t num_this_rank = 0;
  
  // stuff to condense preorder labels
  uint64_t* pre_per_rank = new uint64_t[nprocs];
  uint64_t* pre_offsets_rank = new uint64_t[nprocs+1];
  uint64_t* pre_per_thread = NULL;
  uint64_t* pre_offsets_thread = NULL;
  uint64_t rank_min_preorder = NULL_KEY;
  uint64_t rank_max_preorder = 0;
  uint64_t rank_preorder_spread = 0;
  uint64_t spread_per_thread = 0;
  
  comm->global_queue_size = 1;
#pragma omp parallel
{
  thread_queue_t tq;
  pre_thread_data_t pret;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_pre(&pret);
  init_thread_comm(&tc);
  int32_t tid = omp_get_thread_num();
  int32_t nt = omp_get_num_threads();
  
#pragma omp for
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    num_children[vert_index] = 0;
  
  // every vertex constructs for each child array of nexts, ranks, counts
  // Leafs, single to parent
  // parents, num children + 1 to parent
  // all counts to 1

  // first, count up children
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    visited[i] = false;

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert_index = i;
    uint64_t vert = g->local_unmap[i];
    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    for (uint64_t j = 0; j < out_degree; ++j) {
      if (parents[outs[j]] == vert && !visited[outs[j]]) {
        visited[outs[j]] = true;
        ++num_children[vert_index];
      }
    }
    
    // printf("vert: %li, childs %li, num_children %li\n",
    //   vert, childs, num_children[i]);
  }
  
  // determine what subscript each vertex is for their parent
  // parents communicate to each vertex their subscript id
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    visited[i] = false;
  
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert_index = i;
    uint64_t vert = g->local_unmap[i];
    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    uint64_t subscript = 1;
    
    for (uint64_t j = 0; j < out_degree; ++j) {
      if (parents[outs[j]] == vert && !visited[outs[j]]) {
        visited[outs[j]] = true;
        uint64_t child_index = outs[j];
        if (child_index < g->n_local) {
          uint64_t child = g->local_unmap[child_index];
          add_vid_to_queue(&tq, q, child, subscript);
        } else {
          add_vid_to_send(&tq, q, child_index, subscript);
        }
        subscript++;
      }
    }
  }
  
  empty_queue(&tq, q);
  empty_send(&tq, q);
#pragma omp barrier
      
#pragma omp single
{
  printf("Q next: %lu -- Q send: %lu\n", q->next_size, q->send_size);
  exchange_verts_bicc(g, comm, q);
}

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    subscripts[i] = NULL_KEY;
  
  // set the subscripts after the exchange
#pragma omp for 
  for (uint64_t i = 0; i < q->queue_size; i += 2) {
    uint64_t vert = q->queue[i];
    uint64_t vert_index = get_value(g->map, vert);
    uint64_t subscript = q->queue[i+1];
    assert(vert_index < g->n_local);
    subscripts[vert_index] = subscript;
  }     

#pragma omp for 
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (subscripts[i] == NULL_KEY)
      printf("BVA %lu\n", i);
  
  
  // next, initialize array for nexts, next_subs, ranks, counts
#pragma omp for schedule(guided)
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    nexts[vert_index] = new uint64_t[num_children[vert_index]+1];
    next_subs[vert_index] = new uint64_t[num_children[vert_index]+1];
    counts[vert_index] = new uint64_t[num_children[vert_index]+1];
    ranks[vert_index] = new uint64_t[num_children[vert_index]+1];
        
    for (uint64_t i = 0; i < num_children[vert_index]; ++i) {
      nexts[vert_index][i] = NULL_KEY;      // where we're pointing
      next_subs[vert_index][i] = NULL_KEY;  // subscript to where we point
      counts[vert_index][i] = 1;            // how many hops to next
      ranks[vert_index][i] = NULL_KEY;      // rank of next
    }
    // last item always points to parent (or self if root)
    uint64_t parent = parents[vert_index];
    uint64_t parent_index = get_value(g->map, parent);
    uint64_t parent_rank = -1;
    if (parent_index < g->n_local)
      parent_rank = (uint64_t)procid;
    else
      parent_rank = (uint64_t)g->ghost_tasks[parent_index - g->n_local];
    
    nexts[vert_index][num_children[vert_index]] = parent;
    next_subs[vert_index][num_children[vert_index]] = subscripts[vert_index];
    counts[vert_index][num_children[vert_index]] = 1;
    ranks[vert_index][num_children[vert_index]] = parent_rank;
    
    // handle the root case, aka terminator
    uint64_t vert = g->local_unmap[vert_index];
    if (parent == vert) {
      counts[vert_index][num_children[vert_index]] = 0;
      next_subs[vert_index][num_children[vert_index]] = num_children[vert_index];
    }
  }
  
  // finish building arrays for children
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    visited[i] = false;
  
#pragma omp for schedule(guided)
  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert_index = i;
    uint64_t vert = g->local_unmap[i];
    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    uint64_t childs = 0;
    
    for (uint64_t j = 0; j < out_degree; ++j) {
      if (parents[outs[j]] == vert && !visited[outs[j]]) {
        visited[outs[j]] = true;
        uint64_t child_index = outs[j];
        uint64_t subscript = childs++;
        uint64_t child = NULL_KEY;
        int32_t child_rank = -1;
        if (child_index < g->n_local) {
          child_rank = procid;
          child = g->local_unmap[child_index];
        } else {
          child_rank = g->ghost_tasks[child_index - g->n_local];
          child = g->ghost_unmap[child_index - g->n_local];
        }
        
        nexts[vert_index][subscript] = child;
        next_subs[vert_index][subscript] = 0;
        counts[vert_index][subscript] = 1;
        ranks[vert_index][subscript] = child_rank;
      }
    }
  }
  
  
  // initialize data to pass
  // {vid, subscript, rank, next, next_subscript, rank_next}
#pragma omp for 
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    uint64_t vert = g->local_unmap[vert_index];

    // initialize passing to children
    for (uint64_t s = 0; s < num_children[vert_index]; ++s) {
      uint64_t subscript = s;                   // where to come back and update
      uint64_t next = nexts[vert_index][s];     // child vid
      int32_t next_rank = ranks[vert_index][s]; // child rank
      uint64_t next_subscript = 0;              // start at sub 0 for children
        
      add_to_pre(&pret, preq,
        vert, procid, subscript,
        next, next_rank, next_subscript, 1);
    }
    
    // initialize passing to parent
    uint64_t subscript = num_children[vert_index];
    uint64_t next = nexts[vert_index][num_children[vert_index]];
    int32_t next_rank = ranks[vert_index][num_children[vert_index]];
    uint64_t next_subscript = subscripts[vert_index];
    uint64_t count = counts[vert_index][num_children[vert_index]];
    
    // don't add the terminator
    if (count > 0) {
      add_to_pre(&pret, preq,
        vert, procid, subscript,
        next, next_rank, next_subscript, 1);
    }
  }
 
  empty_pre_queue(&pret, preq);
#pragma omp barrier

  // while still passing
  while (comm->global_queue_size)
  {
    if (debug && pret.tid == 0) { 
      printf("Rank: %d preorder_tree() GQ: %lu, PQ: %lu\n", 
        procid, comm->global_queue_size, preq->queue_size); 
    }        
    
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
    MPI_Allreduce(&comm->total_recv, &comm->global_queue_size, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    preq->queue_size = comm->total_recv;
    preq->next_size = 0;
} // end single

  // update queue data
  // {vid, rank, subscript, 
  // next = nexts[next], rank_next = ranks[next], sub_next = subs[next],
  // count += counts[next]}
#pragma omp for 
    for (uint64_t i = 0; i < preq->queue_size; i+=7) {
      uint64_t vert = preq->queue[i];
      uint64_t rank = preq->queue[i+1];
      uint64_t subscript = preq->queue[i+2];
      uint64_t next = preq->queue[i+3];
      uint64_t next_rank = preq->queue[i+4];
      uint64_t next_sub = preq->queue[i+5];
      uint64_t count = preq->queue[i+6];
      
      uint64_t next_index = get_value(g->map, next);
      //printf("OLD COUNT %lu %lu\n", vert, count);
      
      // check if we're at the end
      if (next_sub == NULL_KEY) {
        printf("fucked %d %lu %lu %lu %lu %lu %lu %lu\n", procid,
          vert, rank, subscript, next, next_rank, next_sub, count);
      }
      if (counts[next_index][next_sub] == 0) {
        next = NULL_KEY;
        next_rank = NULL_KEY;
        next_sub = NULL_KEY;
      } else {
        next = nexts[next_index][next_sub];
        next_rank = ranks[next_index][next_sub];
        count += counts[next_index][next_sub];
        next_sub = next_subs[next_index][next_sub];
      }
      // if (next == 3950)
      //   printf("BALLS %lu %lu %lu %lu %lu\n", 
      //     next, next_rank, next_sub, count);
      
      //printf("New COUNT %lu %lu\n", vert, count);
      
      // we're going to send this back to vert, so it can update its arrays
      add_to_pre(&pret, preq,
        next, next_rank, next_sub, 
        vert, rank, subscript, count);
    }
    
    empty_pre_queue(&pret, preq);
#pragma omp barrier
    
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
    MPI_Allreduce(&comm->total_recv, &comm->global_queue_size, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    preq->queue_size = comm->total_recv;
    preq->next_size = 0;
} // end single
    
    // update pointer info then send package to jump
#pragma omp for 
    for (uint64_t i = 0; i < preq->queue_size; i+=7) {      
      uint64_t next = preq->queue[i];
      uint64_t next_rank = preq->queue[i+1];
      uint64_t next_sub = preq->queue[i+2];
      uint64_t vert = preq->queue[i+3];
      uint64_t rank = preq->queue[i+4];
      uint64_t subscript = preq->queue[i+5];
      uint64_t count = preq->queue[i+6];
      
      uint64_t vert_index = get_value(g->map, vert);
      
      // update our pointer info      
      // and send the package on its way if it hasn't reached the end
      if (next_rank != NULL_KEY) {
        nexts[vert_index][subscript] = next;
        next_subs[vert_index][subscript] = next_sub;
        counts[vert_index][subscript] = count;
        ranks[vert_index][subscript] = next_rank;
        add_to_pre(&pret, preq,
          vert, rank, subscript,
          next, next_rank, next_sub, count);
      }
    }
    
    empty_pre_queue(&pret, preq);
#pragma omp barrier
  } // end while  
  
  // get max preorder
#pragma omp for reduction(max:max_preorder)
  for (uint64_t i = 0; i < g->n_local; ++i) {
    if (counts[i][0] > max_preorder) {
      max_preorder = counts[i][0];
    }
  }
  
#pragma omp single
{
  printf("Rank %d max preorder %lu\n", procid, max_preorder);
  MPI_Allreduce(MPI_IN_PLACE, &max_preorder, 1, 
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  printf("Rank %d global preorder %lu\n", procid, max_preorder);
}
  
  // set values using max preorder
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i) {
    preorders[i] = max_preorder - counts[i][0];
  }

#pragma omp single
{
  num_per_rank = max_preorder / (uint64_t)nprocs + 1;  
}
  // build arrays and do exchange
#pragma omp for
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    uint64_t vert = g->local_unmap[vert_index];
    uint64_t preorder = preorders[vert_index];    
    int32_t rank = preorder / num_per_rank;
    
    // terribly inefficient
    add_to_pre(&pret, preq,
          (uint64_t)procid, 0, 0,
          vert, rank, preorder, 0);
  }
  
  empty_pre_queue(&pret, preq);
#pragma omp barrier
    
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
  MPI_Allreduce(&comm->total_recv, &comm->global_queue_size, 1,
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  preq->queue_size = comm->total_recv;
  preq->next_size = 0;
  
  
  // share how many per rank and compute prefix sums
  for (int32_t i = 0; i < nprocs; ++i)
    pre_per_rank[i] = 0;
  
  num_this_rank = preq->queue_size / 7;
  printf("Rank %d num %lu\n", procid, num_this_rank);
  
  MPI_Allgather(&num_this_rank, 1, MPI_UINT64_T, 
    pre_per_rank, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  pre_offsets_rank[0] = 0;
  for (int32_t i = 1; i < nprocs; ++i) {
    pre_offsets_rank[i] = pre_offsets_rank[i-1] + pre_per_rank[i-1];
    printf("Rank %d offsets rank %d %lu -- %lu %lu\n", procid, i,  
      pre_offsets_rank[i], pre_offsets_rank[i-1], pre_per_rank[i-1]);
  }
} // end single

  // do parallel quicksort
  // first, get min/max offsets
#pragma omp for reduction(max:rank_max_preorder) \
                      reduction(min:rank_min_preorder)
  for (uint64_t i = 0; i < preq->queue_size; i += 7) {
    uint64_t preorder = preq->queue[i+5];
    if (preorder > rank_max_preorder)
      rank_max_preorder = preorder;
    if (preorder < rank_min_preorder)
      rank_min_preorder = preorder;
  }

#pragma omp single
{
  printf("Rank %d max preorder %lu min preorder %lu\n",
    procid, rank_max_preorder, rank_min_preorder);
  rank_preorder_spread = rank_max_preorder - rank_min_preorder;
  spread_per_thread = rank_preorder_spread / nt + 1;
  pre_per_thread = (uint64_t*)calloc(nt, sizeof(uint64_t));
  pre_offsets_thread = (uint64_t*)calloc(nt+1, sizeof(uint64_t));
}
  
  // count how large each thread's array will be
  // get start<->end for the preorders I'm considering
  uint64_t my_thread_start = (uint64_t)tid*spread_per_thread;
  uint64_t my_thread_end = (uint64_t)(tid+1)*spread_per_thread;
  uint64_t my_pre_count = 0;
  printf("Rank %d thread %d my start %lu my end %lu\n",
    procid, tid, my_thread_start, my_thread_end);
    
  // do counting
  for (uint64_t i = 0; i < preq->queue_size; i += 7) {
    uint64_t preorder = preq->queue[i+5] - rank_min_preorder;
    if (preorder >= my_thread_start && preorder < my_thread_end)
      ++my_pre_count;
  }
  
  // determine thread offsets
  pre_per_thread[tid] = my_pre_count;
  printf("Rank %d thread %d my count %lu\n",
    procid, tid, my_pre_count);
  
#pragma omp barrier
  
#pragma omp single
{
  pre_offsets_thread[0] = 0;
  for (int32_t i = 1; i < nt; ++i) {
    pre_offsets_thread[i] = pre_offsets_thread[i-1] + pre_per_thread[i-1];
    printf("offsets thread %d %lu -- %lu %lu\n", i,  pre_offsets_thread[i],
      pre_offsets_thread[i-1], pre_per_thread[i-1]);
  }
}

  // allocate space
  uint64_t* my_preorders = new uint64_t[my_pre_count*3];
  my_pre_count = 0;
  
  // place values into my thread
  // preorder labels will now be from 0 ... (max-min preorder)
  for (uint64_t i = 0; i < preq->queue_size; i += 7) {
    uint64_t rank = preq->queue[i];
    uint64_t vert = preq->queue[i+3];
    uint64_t preorder = preq->queue[i+5] - rank_min_preorder;
    if (preorder >= my_thread_start && preorder < my_thread_end) {
      my_preorders[my_pre_count++] = preorder;
      my_preorders[my_pre_count++] = vert;
      my_preorders[my_pre_count++] = rank;
    }
  }
  
  printf("Rank %d thread %d my range %lu %lu\n",
    procid, tid, pre_offsets_thread[tid] + pre_offsets_rank[procid],
    pre_offsets_thread[tid] + pre_offsets_rank[procid] + pre_per_thread[tid]);
  
  // do my sorting
  if (my_pre_count > 0) {
    quicksort_inc(my_preorders, 0, my_pre_count-3);

    // do a relabeling using calculated offsets
    // my labels are my ordering+thread_offsets+rank_offset
    // place labels into queue
    for (uint64_t i = 0; i < my_pre_count; i += 3) {
      //printf("before map %lu after map %lu -- %lu %lu %lu\n", my_preorders[i],
      //  i/3 + pre_offsets_thread[tid] + pre_offsets_rank[procid],
      //  i/3, pre_offsets_thread[tid], pre_offsets_rank[procid]);
      my_preorders[i] = i/3 + 
                          pre_offsets_thread[tid] + pre_offsets_rank[procid];
      uint64_t preorder = my_preorders[i];
      uint64_t vert = my_preorders[i+1];
      uint64_t rank = my_preorders[i+2];
      add_to_pre(&pret, preq,
            0, 0, 0,
            vert, rank, preorder, 0);
    }
      
    empty_pre_queue(&pret, preq);
  }
#pragma omp barrier
    
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
  memcpy(preq->queue, comm->recvbuf_vert, comm->total_recv*sizeof(uint64_t));
  clear_recvbuf_pre(comm);
  MPI_Allreduce(&comm->total_recv, &comm->global_queue_size, 1,
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  preq->queue_size = comm->total_recv;
  preq->next_size = 0;
} // end single

#pragma omp for
  for (uint64_t i = 0; i < preq->queue_size; i += 7) {
    uint64_t vert = preq->queue[i+3];
    uint64_t preorder = preq->queue[i+5];
    uint64_t vert_index = get_value(g->map, vert);
    preorders[vert_index] = preorder;
    //printf("%lu %lu %lu\n", vert_index, vert, preorder);
  }

#pragma omp single
{
  uint64_t* all_preorders = new uint64_t[g->n];
  
  for (uint64_t i = 0; i < g->n; ++i)
    all_preorders[i] = 0;
  
  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert = g->local_unmap[i];
    all_preorders[vert] = preorders[i];
  }
  
  MPI_Allreduce(MPI_IN_PLACE, all_preorders, g->n, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  bool* has_label = new bool[g->n];
  for (uint64_t i = 0; i < g->n; ++i)
    has_label[i] = 0;
  
  for (uint64_t i = 0; i < g->n; ++i) {
    //printf("%lu %lu\n", i, all_preorders[i]);
    assert(has_label[all_preorders[i]] == false);
    has_label[all_preorders[i]] = true;
  }
}
  
  // do boundary exchange of preorders  
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
                           vert_index, preorders[vert_index]);
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
    preorders[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_comm(&tc);



  // do an iterative parallel merge sort
//   // https://www.geeksforgeeks.org/iterative-merge-sort/
// #pragma omp for
//   for (uint64_t i = 0; i < preq->queue_size; i+=7) {
//     uint64_t vert = preq->queue[i+3];
//     uint64_t rank = preq->queue[i+4];
//     uint64_t preorder = preq->queue[i+5];
    
// for (uint64_t curr_size = 1; curr_size <= num_this_rank - 1; curr_size *= 2) {
// #pragma omp for
//   for (left_start=0; left_start<n-1; left_start += 2*curr_size)
//   {
//     uint64_t mid = min(left_start + curr_size - 1, num_this_rank - 1);
//     uint64_t right_end = min(left_start + 2*curr_size - 1, num_this_rank - 1);
//     uint64_t n1 = mid - left_start + 1;
//     uint64_t n2 = right_end - mid;
//     uint64_t* L = new uint64_t[n1];
//     uint64_t* R = new uint64_t[n2];
//     memcpy(L, &arr[left_start], n1*sizeof(uint64_t));
//     memcpy(R, &arr[mid + 1], n2*sizeof(uint64_t));
 
//     uint64_t i = 0;
//     uint64_t j = 0;
//     uint64_t k = left_start;
//     while (i < n1 && j < n2) {
//       if (L[i] <= R[j]) {
//         arr[k] = L[i];
//         ++i;
//       } else {
//         arr[k] = R[j];
//         ++j;
//       }
//       ++k;
//     }
 
//     while (i < n1) {
//       arr[k] = L[i];
//       i++;
//       k++;
//     }
 
//     while (j < n2) {
//       arr[k] = R[j];
//       j++;
//       k++;
//     }
//   }
// }
 
  
} // end parallel

  return 0;
}


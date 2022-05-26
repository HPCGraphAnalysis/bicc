
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

dist_graph_t* reduce_graph(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q)
{
  if (debug) { printf("procid %d reduce_graph() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }
  
  graph_gen_data_t* ggi_new = 
      (graph_gen_data_t*)malloc(sizeof(graph_gen_data_t));
  ggi_new->n = g->n;
  ggi_new->m = 0;
  ggi_new->n_local = g->n_local;
  ggi_new->n_offset = g->n_offset;
  ggi_new->m_local_read = 0;
  ggi_new->m_local_edges = 0;
  ggi_new->gen_edges = NULL;
  
  uint64_t* parents1 = new uint64_t[g->n_total];
  uint64_t* levels = new uint64_t[g->n_total];
  spanning_tree(g, comm, q, parents1, levels, g->max_degree_vert);
  
  uint64_t* labels = new uint64_t[g->n_total];
  connected_components(g, comm, q, parents1, labels);
  
  uint64_t* parents2 = new uint64_t[g->n_total];
  spanning_forest(g, comm, q, parents1, parents2, levels, labels);
  delete [] levels;
  delete [] labels;
  
//   ggi_new->gen_edges = (uint64_t*)malloc(sizeof(uint64_t)*g->n_local*8);
// #pragma omp parallel 
// {
//   uint64_t tq[THREAD_H_QUEUE_SIZE];
//   uint64_t tq_size = 0;
  
// #pragma omp for
//   for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
//     uint64_t v = g->local_unmap[vert_index];
//     uint64_t u = parents1[vert_index];
//     uint64_t w = parents2[vert_index];
//     assert(u != NULL_KEY);
//     if (v != u) {
//       add_to_queue(tq, tq_size, ggi_new->gen_edges, ggi_new->m_local_read, v, u);
//       add_to_queue(tq, tq_size, ggi_new->gen_edges, ggi_new->m_local_read, u, v);
//       //printf("edge 1 %lu %lu\n", v, u);
//     }
//     if (w != NULL_KEY && v != w) {
//       add_to_queue(tq, tq_size, ggi_new->gen_edges, ggi_new->m_local_read, v, w);
//       add_to_queue(tq, tq_size, ggi_new->gen_edges, ggi_new->m_local_read, w, v);
//       //printf("edge 2 %lu %lu\n", v, w);
//     }
//   }

//   empty_queue(tq, tq_size, ggi_new->gen_edges, ggi_new->m_local_read);
// } // end parallel
  
  // ggi_new->m_local_read /= 2;
  // ggi_new->m_local_edges = ggi_new->m_local_read;
  // MPI_Allreduce(&ggi_new->m_local_edges, &ggi_new->m, 1, 
  //   MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  dist_graph_t* g_new = (dist_graph_t*)malloc(sizeof(dist_graph_t));
  // if (nprocs > 1) {
  //   exchange_edges(ggi_new, comm);
  //   create_graph(ggi_new, g_new);
  //   relabel_edges(g_new);
  // } else {
  //   create_graph_serial(ggi_new, g_new);
  // }
  // free(ggi_new);
  
  elt = omp_get_wtime() - elt;
  if (debug) printf("Rank %d reduce_graph() success, %lf (s)\n", procid, elt);
  
  return g_new;
}



int spanning_tree(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                   uint64_t* parents, uint64_t* levels, uint64_t root)
{
  if (debug) { printf("procid %d spanning_tree() start\n", procid); }
  
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
      } else {
        printf("EROR %d %lu\n", procid, vert_index);
      }
    }
  }

  if (debug) printf("Rank %d spanning_tree() success\n", procid);

  return 0;
}



int connected_components(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                         uint64_t* parents, uint64_t* labels)
{ 
  if (debug) { printf("Rank %d connected_components() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;
  
  bool* process_vert = (bool*)malloc(g->n_local*sizeof(bool));
  bool* process_vert_next = (bool*)malloc(g->n_local*sizeof(bool));

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
  for (uint64_t vert_index = 0; vert_index < g->n_total; ++vert_index) {
    uint64_t vert = NULL_KEY;
    if (vert_index < g->n_local)
      vert = g->local_unmap[vert_index];
    else
      vert = g->ghost_unmap[vert_index - g->n_local];
    labels[vert_index] = vert;
  }

#pragma omp for
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
    process_vert[vert_index] = true;
    process_vert_next[vert_index] = false;
  }
    
  while (comm->global_queue_size)
  {
    tq.thread_queue_size = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      if (!process_vert[vert_index]) continue;
      process_vert[vert_index] = false;
      
      bool send = false;
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
        
        if (parents[vert_index] == out || parents[out_index] == vert)
          continue;
        else if (labels[out_index] < labels[vert_index])
        {
          labels[vert_index] = labels[out_index];
          send = true;
        }
      }

      if (send) {
        add_vid_to_send(&tq, q, vert_index);
        ++tq.thread_queue_size;
        for (uint64_t j = 0; j < out_degree; ++j) {
          if (outs[j] < g->n_local)
            process_vert_next[outs[j]] = true;
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
                             vert_index, labels[vert_index]);
    }

    empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
    temp_send_size = q->send_size;
    exchange_vert_data(g, comm, q);
} // end single


#pragma omp for schedule(guided)
    for (uint64_t i = 0; i < comm->total_recv; ++i)
    {
      uint64_t vert_index = get_value(g->map, comm->recvbuf_vert[i]);
      assert(vert_index >= g->n_local);
      labels[vert_index] = comm->recvbuf_data[i];
      
      vert_index -= g->n_local;
      uint64_t out_degree = ghost_out_degree(g, vert_index);
      uint64_t* outs = ghost_out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
        process_vert_next[outs[j]] = true; 
    }

#pragma omp single
{   
    clear_recvbuf_vid_data(comm);
    ++level;
    
    bool* tmp = process_vert;
    process_vert = process_vert_next;
    process_vert_next = tmp;

    if (debug) printf("Rank %d send_size %lu global_size %li\n", 
      procid, temp_send_size, comm->global_queue_size);
}

  } // end while
  clear_thread_comm(&tc);
  clear_thread_queue(&tq);
} // end parallel 

  if (verify) {
    uint64_t fuckups = 0;
    uint64_t ghosts = 0;
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      uint64_t vert = g->local_unmap[vert_index];
      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];   
        uint64_t out = NULL_KEY;
        if (out_index < g->n_local)
          out = g->local_unmap[out_index];
        else {
          out = g->ghost_unmap[out_index - g->n_local];
          ++ghosts;
        }
        
        if (parents[vert_index] == out || parents[out_index] == vert)
          continue;
        else if (labels[vert_index] != labels[out_index])
        {
          ++fuckups;
          //assert(labels[vert_index] == labels[out_index]);
        }
      }
    }
    printf("%d Num ghost issues %lu / %lu\n", procid, fuckups, ghosts);
  }

  if (debug) printf("Rank %d connected_components() success, %lf (s)\n", 
    procid, omp_get_wtime() - elt);

  return 0;
}


int spanning_forest(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                    uint64_t* parents_old, uint64_t* parents, uint64_t* levels,
                    uint64_t* labels)
{
  if (debug) { printf("procid %d spanning_forest() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }
  
  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  comm->global_queue_size = 1;
  uint64_t temp_send_size = 0;
  uint64_t level = 1;
  uint64_t num_roots = 0;
  uint64_t num_local_roots = 0;
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
  
  // need to get roots first
#pragma omp for reduction(+:num_local_roots) reduction(+:num_roots)
  for (uint64_t vert_index = 0; vert_index < g->n_total; ++vert_index)
  {
    uint64_t vert = NULL_KEY;
    if (vert_index < g->n_local)
      vert = g->local_unmap[vert_index];
    else
      vert = g->ghost_unmap[vert_index - g->n_local];
    
    if (labels[vert_index] == vert) {
      // we have root
      parents[vert_index] = vert;
      levels[vert_index] = 0;
      ++num_roots;
      if (vert_index < g->n_local)
        ++num_local_roots;
    }
    // if (vert_index >= g->n_local) {
    //   printf("a %d %lu %lu %lu\n", procid, vert_index, vert, labels[vert_index]);
    // }
  }

  if (debug) {
#pragma omp single
    printf("Rank %d local roots %lu roots %lu\n", 
      procid, num_local_roots, num_roots);
  }
  
  while (comm->global_queue_size)
  {
    tq.thread_queue_size = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      if (parents[vert_index] != NULL_KEY)
        continue;

      uint64_t vert = g->local_unmap[vert_index];
      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        if (levels[out_index] == level-1) {
          uint64_t out = NULL_KEY;
          if (out_index < g->n_local)
            out = g->local_unmap[out_index];
          else
            out = g->ghost_unmap[out_index - g->n_local];
          
          if (parents[vert_index] == out || parents[out_index] == vert ||
              parents_old[vert_index] == out || parents_old[out_index] == vert)
            continue;
          else {
            assert(labels[vert_index] == labels[out_index]);
            parents[vert_index] = out;
            levels[vert_index] = level;
            add_vid_to_send(&tq, q, vert_index);
            //add_vid_to_queue(&tq, q, vert_index);
            ++tq.thread_queue_size;
            break;
          }
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
        if (levels[get_value(g->map, parents[vert_index])] != levels[vert_index] - 1 && parents[vert_index] != g->local_unmap[vert_index])
          printf("Mismatch p: %lu pl: %lu - v: %lu vl: %lu\n",
            parents[vert_index], levels[get_value(g->map, parents[vert_index])],
            g->local_unmap[vert_index], levels[vert_index]);
      }
    }
  
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
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
        
        if (parents_old[vert_index] != out && parents_old[out_index] != vert &&
            parents[vert_index] != out && parents[out_index] != vert && labels[out_index] == labels[vert_index])
          printf("bad edge: %lu %lu\n", vert, out);
      }
    }
  }
  
  if (debug) printf("Rank %d spanning_forest() success, %lf (s)\n", 
    procid, omp_get_wtime() - elt);

  return 0;  
}

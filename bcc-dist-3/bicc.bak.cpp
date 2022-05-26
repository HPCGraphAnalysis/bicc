


// int run_bicc(dist_graph_t* g, mpi_data_t* comm, 
//   uint64_t* high, uint64_t* low, uint64_t* parents, 
//   uint64_t* levels, uint64_t* high_levels)
// { 
//   if (debug) { printf("Task %d run_bicc() start\n", procid); }
//   double elt = 0.0;
//   if (verbose) {
//     MPI_Barrier(MPI_COMM_WORLD);
//     elt = omp_get_wtime();
//   }

//   uint64_t color_changes = g->n;
//   uint64_t iter = 0;

//   for (int32_t i = 0; i < nprocs; ++i)
//     comm->sendcounts_temp[i] = 0;

// #pragma omp parallel default(shared)
// {
//   thread_comm_t tc;
//   init_thread_comm(&tc);
  
// #pragma omp for schedule(guided)
//   for (uint64_t i = 0; i < g->n_local; ++i) {
//     update_sendcounts_thread(g, &tc, i, 3);
//   }

//   for (int32_t i = 0; i < nprocs; ++i)
//   {
// #pragma omp atomic
//     comm->sendcounts_temp[i] += tc.sendcounts_thread[i];

//     tc.sendcounts_thread[i] = 0;
//   }
// #pragma omp barrier

// #pragma omp single
// {
//   init_sendbuf_vid_data(comm);    
//   init_recvbuf_vid_data(comm);
// }

// #pragma omp for schedule(guided)
//   for (uint64_t i = 0; i < g->n_local; ++i) {
//     update_vid_data_queues(g, &tc, comm, i, high[i], low[i], high_levels[i]);
//   }

//   empty_vid_data(&tc, comm);
// #pragma omp barrier

// #pragma omp single
// {
//   exchange_verts(comm);
//   exchange_data(comm);
// }

// #pragma omp for
//   for (uint64_t i = 0; i < comm->total_recv; i += 3)
//   {
//     uint64_t vert_index = get_value(g->map, comm->recvbuf_vert[i]);
//     assert(vert_index < g->n_total);
//     high[vert_index] = comm->recvbuf_data[i];
//     low[vert_index] = comm->recvbuf_data[i+1];
//     high_levels[vert_index] = comm->recvbuf_data[i+2];
//     comm->recvbuf_vert[i] = vert_index;
//   }
  
// #pragma omp single
//   if (debug) printf("Task %d initialize ghost data success\n", procid);

// #pragma omp for
//   for (uint64_t i = 0; i < comm->total_send; ++i)
//   {
//     uint64_t index = get_value(g->map, comm->sendbuf_vert[i]);
//     assert(index < g->n_total);
//     comm->sendbuf_vert[i] = index;
//   } 

// #pragma omp single
//   if (debug) printf("Task %d initialize send buff index success\n", procid);

//   while (color_changes)
//   {    
// #pragma omp single
//     if (debug) {
//       printf("Task %d Iter %lu Changes %lu run_bicc(), %9.6lf\n", 
//         procid, iter, color_changes, omp_get_wtime() - elt);
//     }
  
// #pragma omp single
//     color_changes = 0;

// #pragma omp for schedule(guided) reduction(+:color_changes)
//     for (uint64_t i = 0; i < g->n_local; ++i)
//     {
//       uint64_t vert_index = i;
//       uint64_t vert_high = high[vert_index];
      
//       int vert = vert_index;
//       uint64_t vert_high_level = high_levels[vert_index];
//       uint64_t vert_low = low[vert_index];

//       uint64_t out_degree = out_degree(g, vert_index);
//       uint64_t* outs = out_vertices(g, vert_index);
//       for (uint64_t j = 0; j < out_degree; ++j)
//       {
//         uint64_t out_index = outs[j];
//         uint64_t out;
//         if (out_index < g->n_local)
//           out = g->local_unmap[out_index];
//         else
//           out = g->ghost_unmap[out_index-g->n_local];
//         uint64_t out_high = high[out_index];
//         uint64_t out_high_level = high_levels[out_index];
//         if (vert_high == out)
//           continue;

//         if (out_high_level == vert_high_level && out_high > vert_high)
//         {
//           high[vert_index] = out_high;
//           vert_high = out_high;
//           high_levels[vert_index] = out_high_level;
//           vert_high_level = out_high_level;
//           ++color_changes;
//         }
//         if (out_high_level < vert_high_level)
//         {
//           high[vert_index] = out_high;
//           vert_high = out_high;
//           high_levels[vert_index] = out_high_level;
//           vert_high_level = out_high_level;
//           ++color_changes;   
//         }
//         if (vert_high == out_high)
//         {
//           uint64_t out_low = low[out_index];
//           if (out_low < vert_low)
//           {
//             low[vert_index] = out_low;
//             vert_low = out_low;
//             ++color_changes;
//           }
//         }
//       }
//     }

// #pragma omp for
//     for (uint64_t i = 0; i < comm->total_send; i += 3) {
//       comm->sendbuf_data[i] = high[comm->sendbuf_vert[i]];
//       comm->sendbuf_data[i+1] = low[comm->sendbuf_vert[i]];
//       comm->sendbuf_data[i+2] = high_levels[comm->sendbuf_vert[i]];
//     }

// #pragma omp single
// {
//     exchange_data(comm);
// }

// #pragma omp for
//     for (uint64_t i = 0; i < comm->total_recv; i += 3) {
//       high[comm->recvbuf_vert[i]] = comm->recvbuf_data[i];
//       low[comm->recvbuf_vert[i]] = comm->recvbuf_data[i+1];
//       high_levels[comm->recvbuf_vert[i]] = comm->recvbuf_data[i+2];
//     }

// #pragma omp single
// {
//   MPI_Allreduce(MPI_IN_PLACE, &color_changes, 1, 
//     MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
//   ++iter;
// }
//   } // end for loop

//   clear_thread_comm(&tc);
// } // end parallel

//   clear_allbuf_vid_data(comm);

//   if (verbose) {
//     elt = omp_get_wtime() - elt;
//     printf("Task %d, run_bicc() time %9.6f (s)\n", procid, elt);
//   }
//   if (debug) { printf("Task %d run_bicc() success\n", procid); }

//   return 0;
// }


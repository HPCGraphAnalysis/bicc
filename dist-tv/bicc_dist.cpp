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
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <utility>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <cstring>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include "bicc_dist.h"

#include "comms.h"
#include "dist_graph.h"
//#include "bfs.h"
#include "reduce_graph.h"

#define NOT_VISITED 18446744073709551615U
#define VISITED 18446744073709551614U
#define ASYNCH 1

int bicc_bfs(dist_graph_t* g, mpi_data_t* comm, 
  uint64_t* parents, uint64_t* levels, uint64_t root)
{  
  if (debug) { printf("procid %d bicc_bfs() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  queue_data_t* q = (queue_data_t*)malloc(sizeof(queue_data_t));;
  init_queue_data(g, q);

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  uint64_t root_index = get_value(g->map, root);
  if (root_index != NULL_KEY && root_index < g->n_local)    
  {
    q->queue[0] = root;
    q->queue[1] = root;
    q->queue_size = 2;
  }
  
  bool* visited = new bool[g->n_total];

  uint64_t level = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  init_thread_queue(&tq);

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    parents[i] = NOT_VISITED;

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    levels[i] = NOT_VISITED;
  
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    visited[i] = false;
#pragma omp for
  for(int i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  while (comm->global_queue_size)
  {
#pragma omp single
    if (debug) { 
      printf("Task: %d bicc_bfs() GQ: %lu, TQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; i += 2)
    {
      uint64_t vert = q->queue[i];
      uint64_t parent = q->queue[i+1];
      uint64_t vert_index = get_value(g->map, vert);
      if (parents[vert_index] != VISITED && parents[vert_index] != NOT_VISITED)
        continue;
      
      parents[vert_index] = parent;
      levels[vert_index] = level;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);

      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
   
        uint64_t test = 0;
#pragma omp atomic capture
        { test = parents[out_index]; parents[out_index] = VISITED; }
        
        if (test == NOT_VISITED) {
          if (out_index < g->n_local) {
            add_vid_to_queue(&tq, q, g->local_unmap[out_index], vert);
          } else {
            add_vid_to_send(&tq, q, out_index, vert);
          }
        } else if (test != VISITED) {
          parents[out_index] = test;
        }
      }
    }  

    empty_queue(&tq, q);
    empty_send(&tq, q);
#pragma omp barrier

#pragma omp single
    {
      exchange_verts_bicc(g, comm, q);
      ++level;
    }
  } // end while




  // do full boundary exchange of parents data

  thread_comm_t tc;
  init_thread_comm(&tc);
  
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
                           vert_index, parents[vert_index]);
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
    parents[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_comm(&tc);


// #pragma omp for nowait
//   for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
//     uint64_t vert = g->local_unmap[vert_index];
//     uint64_t parent = parents[vert_index];
//     uint64_t parent_index = get_value(g->map, parent);
    
//     if (parent_index >= g->n_local) {
//       //printf("woo %lu %lu\n", parent_index, vert);
//       add_vid_to_send(&tq, q, parent_index, vert);
//     }
//   }
  
//   empty_queue(&tq, q);
//   empty_send(&tq, q);
// #pragma omp barrier

// #pragma omp single
// {
//   //printf("%lu WOO \n", q->queue_next);
//   exchange_verts_bicc(g, comm, q);
// }

// #pragma omp for
//   for (uint64_t i = 0; i < q->queue_size; i += 2) {
//     uint64_t parent = q->queue[i];
//     uint64_t child = q->queue[i+1];
//     uint64_t child_index = get_value(g->map, child);
//     parents[child_index] = parent;
//     assert(child_index >= g->n_local);
//     //printf("WOO %lu %lu\n", parent, child);
//   }

// #pragma omp for
//   for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index) {
//     uint64_t out_degree = out_degree(g, vert_index);
//     uint64_t* outs = out_vertices(g, vert_index);
//     for (uint64_t j = 0; j < out_degree; ++j) {
//       if (outs[j] >= g->n_local)
//         assert(parents[outs[j]] < VISITED);
//     }
//   }


  clear_thread_queue(&tq);
} // end parallel

  clear_queue_data(q);
  free(q);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d bicc_bfs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d bicc_bfs() success\n", procid); }

  return 0;
}


struct pair_hash{
  std::size_t operator()(const std::pair<uint64_t, uint64_t>& k) const {
    return std::hash<uint64_t>()(k.first) ^ (std::hash<uint64_t>()(k.second)<<1);
  }
};


int procid, nprocs;
int seed;
bool verbose, debug, verify;

extern "C" int bicc_dist_run(dist_graph_t* g)
{
  mpi_data_t comm;
  queue_data_t q;
  init_comm_data(&comm);
  init_queue_data(g, &q);

  bicc_dist(g, &comm, &q);

  clear_comm_data(&comm);
  clear_queue_data(&q);

  return 0;
}

void owned_to_ghost_value_comm(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* values){
  //communicate processors with ghosts back to their owners
  //messages this round consists of gids and a default value that will be filled in
  //by the owning processor, process IDs are implicitly recorded by the alltoallv
  int* sendcnts = new int[nprocs];
  for( int i = 0; i < nprocs; i++){
    sendcnts[i] = 0;
  }

  for(int i = g->n_local; i < g->n_total; i++){
    sendcnts[g->ghost_tasks[i-g->n_local]] += 2;
  }

  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) recvcnts[i] = 0;
  int status = MPI_Alltoall(sendcnts,1, MPI_INT,recvcnts,1,MPI_INT,MPI_COMM_WORLD);
  
  int* sdispls = new int[nprocs];
  int* rdispls = new int[nprocs];
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
  }

  int sendsize = 0;
  int recvsize = 0;
  int* sentcount = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendsize += sendcnts[i];
    recvsize += recvcnts[i];
    sentcount[i] = 0;
  }
  
  int* sendbuf = new int[sendsize];
  int* recvbuf = new int[recvsize];
  
  //go through all the ghosted vertices, send their GIDs to the owning processor
  for(int i = g->n_local; i < g->n_total; i++){
    int proc_to_send = g->ghost_tasks[i - g->n_local];
    int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send];
    sentcount[proc_to_send] += 2;
    sendbuf[sendbufidx++] = g->ghost_unmap[i - g->n_local];
    sendbuf[sendbufidx++] = values[i];
  }
  status = MPI_Alltoallv(sendbuf,sendcnts,sdispls,MPI_INT,recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);

  //fill in the values that were sent along with the GIDs
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx+= 2){
    int gid = recvbuf[exchangeIdx];
    int lid = get_value(g->map, gid);
    if(values[lid] == 0 && recvbuf[exchangeIdx+1] != 0) values[lid] = recvbuf[exchangeIdx+1];
    recvbuf[exchangeIdx + 1] = values[lid];
  }

  
  //communicate owned values back to ghosts
  status = MPI_Alltoallv(recvbuf,recvcnts,rdispls,MPI_INT, sendbuf,sendcnts,sdispls,MPI_INT,MPI_COMM_WORLD);

  for(int updateIdx = 0; updateIdx < sendsize; updateIdx+=2){
    int gid = sendbuf[updateIdx];
    int lid = get_value(g->map, gid);
    values[lid] = sendbuf[updateIdx+1];
  }
  delete [] sendcnts;
  delete [] recvcnts;
  delete [] sdispls;
  delete [] rdispls;
  delete [] sentcount;
  delete [] sendbuf;
  delete [] recvbuf;
}


void communicate_preorder_labels(dist_graph_t* g, std::queue<uint64_t> &frontier, uint64_t* preorder, uint64_t sentinel){
  //communicate ghosts to owners and owned to ghosts
  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendcnts[i] = 0;
  }

  for(int i = g->n_local; i < g->n_total; i++){
    sendcnts[g->ghost_tasks[i-g->n_local]] += 2;
  }

  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) recvcnts[i] = 0;
  int status = MPI_Alltoall(sendcnts,1,MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);
  
  int* sdispls = new int[nprocs];
  int* rdispls = new int[nprocs];
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
  }
  
  int sendsize = 0;
  int recvsize = 0;
  int* sentcount = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendsize += sendcnts[i];
    recvsize += recvcnts[i];
    sentcount[i] = 0;
  }

  int* sendbuf = new int[sendsize];
  int* recvbuf = new int[recvsize];
  
  //send along any information available on the current process
  for(int i = g->n_local; i < g->n_total; i++){
    int proc_to_send = g->ghost_tasks[i-g->n_local];
    int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send];
    sentcount[proc_to_send] += 2;
    sendbuf[sendbufidx++] = g->ghost_unmap[i-g->n_local];
    sendbuf[sendbufidx++] = preorder[i];
  }
  status = MPI_Alltoallv(sendbuf, sendcnts,sdispls,MPI_INT, recvbuf,recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);
  
  //fill in nonzeros that were sent with GIDs
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx += 2){
    int gid = recvbuf[exchangeIdx];
    int lid = get_value(g->map, gid);
    //std::cout<<"Rank "<<procid<<" received preorder label "<<recvbuf[exchangeIdx+1]<<" for vertex "<<gid<<"\n";
    if(preorder[lid] == sentinel && recvbuf[exchangeIdx + 1] != sentinel){
      preorder[lid] = recvbuf[exchangeIdx+1];
      frontier.push(lid);
      //std::cout<<"Rank "<<procid<<" pushing vertex "<<gid<<" onto the local queue\n";
    } //else if (preorder[lid] != 0 && recvbuf[exchangeIdx+1]!=0 && preorder[lid] != recvbuf[exchangeIdx+1]) std::cout<<"*********sent and owned preorder labels mismatch*************\n";
    recvbuf[exchangeIdx+1] = preorder[lid];
  }

  status = MPI_Alltoallv(recvbuf,recvcnts,rdispls,MPI_INT,sendbuf,sendcnts,sdispls,MPI_INT,MPI_COMM_WORLD);
  
  for(int updateIdx=0; updateIdx < sendsize; updateIdx += 2){
    int gid = sendbuf[updateIdx];
    int lid = get_value(g->map, gid);
    //std::cout<<"Rank "<<procid<<" updating vertex "<<gid<<" with label "<<sendbuf[updateIdx+1]<<"\n";
    preorder[lid] = sendbuf[updateIdx+1];
  }
  delete [] sendcnts;
  delete [] recvcnts;
  delete [] sdispls;
  delete [] rdispls;
  delete [] sentcount;
  delete [] sendbuf;
  delete [] recvbuf;

  //std::cout<<"Rank "<<procid<<"'s frontier size = "<<frontier.size()<<"\n";
}

void preorder_label_recursive(dist_graph_t* g,uint64_t* parents, uint64_t* preorder, uint64_t currVtx, uint64_t& preorderIdx){
  preorder[currVtx] = preorderIdx;
  int out_degree = out_degree(g, currVtx);
  uint64_t* outs = out_vertices(g, currVtx);
  for(int i = 0; i < out_degree; i++){
    uint64_t nbor = outs[i];
    if(parents[nbor] == currVtx && preorder[nbor] == NULL_KEY){
      preorderIdx++;
      preorder_label_recursive(g,parents,preorder,nbor,preorderIdx);
    }
  }
}

extern "C" void calculate_descendants(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* parents, uint64_t* n_descendants){
  std::cout<<"Calculating Descendants for all "<<g->n<<" vertices\n";
  for(int i = 0; i < g->n_total; i++) n_descendants[i] = NULL_KEY;
  std::queue<uint64_t> primary_frontier;
  std::queue<uint64_t> secondary_frontier;
  std::queue<uint64_t>* currQueue = &primary_frontier;
  std::queue<uint64_t>* otherQueue = &secondary_frontier;
  for(int i = 0; i < g->n_local; i++){
    primary_frontier.push(i);
  }
  int global_done = 0;
  while (!global_done){
    while(currQueue->size() > 0){
      uint64_t currVert = currQueue->front();
      //std::cout<<"Rank "<<procid<<" is processing vertex "<<g->local_unmap[currVert]<<"\n";
      currQueue->pop();
      if(n_descendants[currVert] != NULL_KEY) continue;
      int out_degree = out_degree(g,currVert);
      int children = 0;
      int children_computed = 0;
      uint64_t n_desc = 0;
      uint64_t* outs = out_vertices(g, currVert);
      std::unordered_set<uint64_t> visited_nbors;
      for(int i = 0; i < out_degree; i++){
        uint64_t nbor_local = outs[i];
	if(visited_nbors.count(nbor_local) > 0) continue;
        if(parents[nbor_local] == g->local_unmap[currVert]){
          children++;
          if(n_descendants[nbor_local] != NULL_KEY){
            children_computed++;
            n_desc+=n_descendants[nbor_local];
          }
        }
	visited_nbors.insert(nbor_local);
      }
      if(children_computed == children){
	/*if(currVert == 89194 || currVert == 471983) std::cout<<"Vertex "<<currVert<<" has "<<children<<"children and they have "<<n_desc<<" descedants, collectively\n";*/
        n_descendants[currVert] = n_desc + children_computed;
        //if(get_value(g->map, parents[currVert]) < g->n_local){
        //  otherQueue->push(get_value(g->map, parents[currVert]));
        //}
      } else {
        //std::cout<<"Rank "<<procid<<" was unable to compute n_desc for vertex "<<g->local_unmap[currVert]<<"\n";
        otherQueue->push(currVert);
      }
    }
    communicate_preorder_labels(g,*otherQueue, n_descendants, NULL_KEY);
    int local_done = otherQueue->size();
    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    global_done = !global_done;
    std::swap(currQueue,otherQueue);
    //printf("Rank %d currQueue size = %d, otherQueue size = %d\n",procid, currQueue->size(),otherQueue->size());
  }
  //std::set<uint64_t> unfinished;
  //all leaves have 1 descendant    
  /*for(uint64_t i = 0; i < g->n_total; i++){
    if(is_leaf[i]) {
      n_descendants[i] = 1;
      //unfinished.insert(parents[i]);
    } else {
      n_descendants[i] = 0;
    }
  }
  std::cout<<"\tDone with initialization\n";
  //see if any leaves' parents can be calculated
  uint64_t unfinished = g->n_total;
  uint64_t last_unfinished = g->n_total+1;
  int local_done = 0;
  int global_done = 0;
  while(!global_done){
    //while local progress is being made
    while (unfinished < last_unfinished){
      last_unfinished = unfinished;
      unfinished = 0;
      for(int parent = 0; parent < g->n_local; parent++){
        if(n_descendants[parent] !=0) continue;
        std::cout<<"Rank "<<procid<<"\tchecking if vertex "<<parent<<" can be calculated\n"; 
        //look at neighbors whose parent is *iter, if all > 0, we have a winner!
        int out_degree = out_degree(g, parent);
        int children = 0;
        int children_computed = 0;
        int n_desc = 1;
        uint64_t* outs = out_vertices(g, parent);
        for(int i = 0; i < out_degree; i++){
          int nbor_local = outs[i];
          //only descendants get counted
          if(parents[nbor_local] == g->local_unmap[parent]){
            children++;
            if(n_descendants[nbor_local] > 0){
              std::cout<<"\t\t"<<nbor_local<<" has "<<n_descendants[nbor_local]<<" descendants\n";
              children_computed++;
              n_desc += n_descendants[nbor_local];
            } else {
              std::cout<<"\t\t"<<nbor_local<<" has "<<n_descendants[nbor_local]<<" descendants\n";
            }
          }
        }
        if(children_computed == children){
          std::cout<<"Successfully computed descendants, vertex "<<parent<<" has "<<n_desc<<" descendants\n";
          n_descendants[parent] = n_desc;
        } else {
          unfinished++;
        }
        
      }
    
    }
    //all_reduce to check if everyone's done
    local_done = last_unfinished == 0;
    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
    //if not done, ghost->owned->ghost exchanges (need to think more about what type of operation needs done with each)
      //break;
    owned_to_ghost_value_comm(g,comm,q,n_descendants);
    

  }
  return 0;*/
}

void calculate_preorder(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* levels,  uint64_t* parents, uint64_t* n_desc, uint64_t* preorder){
  std::cout<<"\tCalculating Preorder Labels\n";
  std::queue<uint64_t> frontier;
  for(int i = 0; i < g->n_local; i++){
    if(levels[i] == 0){
      preorder[i] = 1;
      frontier.push(i);
    }
  }
  int global_done = 0;
  while(!global_done){
    while(frontier.size() > 0){
      uint64_t currVert = frontier.front();
      frontier.pop();
      //std::cout<<"Rank "<<procid<<" is processing vertex "<<g->local_unmap[currVert]<<"\n";
      int out_degree = out_degree(g,currVert);
      uint64_t* outs = out_vertices(g,currVert);
      uint64_t child_label = preorder[currVert]+1;
      std::unordered_set<uint64_t> visited_nbors;
      for(int nbor = 0; nbor < out_degree; nbor++){
        uint64_t nborVert = outs[nbor];
	if(visited_nbors.count(nborVert) > 0) continue;
        if(parents[nborVert] == g->local_unmap[currVert]){
          preorder[nborVert] = child_label;
          child_label += n_desc[nborVert]+1;
          if(nborVert < g->n_local) frontier.push(nborVert);
        }
	visited_nbors.insert(nborVert);
      }
    }
    
    for(int i = 0; i < g->n_total; i++){
      uint64_t global = 0;
      if(i < g->n_local) global = g->local_unmap[i];
      else global = g->ghost_unmap[i-g->n_local];
      //std::cout<<"Rank "<<procid<<"'s vertex "<<global<<" has label "<<preorder[i]<<"\n";
    }
    communicate_preorder_labels(g,frontier,preorder,0);
    int local_done = frontier.size();
    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //std::cout<<"global number of vertices in frontiers: "<<global_done<<"\n";
    global_done = !global_done;
  }
  //go level-by-level on each process, communicating after each level is completed.
  /*int max_level = 0;
  for(int i = 0; i < g->n_local; i++){
    if(max_level < levels[i]) max_level = levels[i];
  }
  int global_max_level = 0;
  MPI_Allreduce(&max_level,&global_max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::cout<<"\t\tRank "<<procid<<"'s Max level = "<<global_max_level<<"\n";
  
  for(int level = 0; level <= global_max_level; level++){
    std::cout<<"Rank "<<procid<<" Calculating labels for level "<<level<<"\n";
    //look through all owned vertices and calculate all their children serially
    for(int i = 0; i < g->n_local; i++){
      if(levels[i] != level) continue;  //either not involved in current calculation, or not able to be calculated yet
      if(level == 0) {                   //base case, root is labeled 1
        preorder[i] = 1;
        std::cout<<"Rank "<<procid<<" setting vertex "<<g->local_unmap[i]<<"'s preorder label to 1\n";
      }
      int out_degree = out_degree(g, i);
      uint64_t* outs = out_vertices(g, i);
      uint64_t child_label = preorder[i] + 1; //first child's label is the parent's plus 1
      for(int j = 0; j < out_degree; j++){
        if(parents[outs[j]] == g->local_unmap[i]){
          preorder[outs[j]] = child_label;
          if(outs[j] < g->n_local){
            std::cout<<"Rank "<<procid<<" setting vertex "<<g->local_unmap[outs[j]]<<"'s preorder label to "<<preorder[outs[j]]<<"\n";
          } else {
            std::cout<<"Rank "<<procid<<" setting vertex "<<g->ghost_unmap[outs[j]-g->n_local]<<"'s preorder label to "<<preorder[outs[j]]<<"\n"; 
          }
          child_label += n_desc[outs[j]]; //the subsequent children need to factor in the previous childs' descendants
        }
      }
    }
    std::cout<<"Rank "<<procid<<": entering communication\n"; 
    owned_to_ghost_value_comm(g,comm,q,preorder);
  }*/
  
}

void calculate_high_low(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* highs, uint64_t* lows, uint64_t* parents, uint64_t* preorder){
  
  for(int i = 0; i < g->n_total; i++){
    highs[i] = 0;
    lows[i] = 0;
  }
  
  std::queue<uint64_t> primary_frontier;
  std::queue<uint64_t> secondary_frontier;
  std::queue<uint64_t>* currQueue = &primary_frontier;
  std::queue<uint64_t>* otherQueue = &secondary_frontier;
  std::unordered_set<uint64_t> verts_in_queue; 
  //put all owned leaves on a frontier
  for(int i = 0; i < g->n_local; i++){
    primary_frontier.push(i);
    verts_in_queue.insert(i);
  }
  int global_done = 0;
  //while the number of vertices in the global queues is not zero
  while(!global_done){
    //while the primary queue is not empty
    //std::cout<<"global_done = "<<global_done<<"\n";
    while(currQueue->size() > 0){
      //std::cout<<"curr_queue.size() = "<<currQueue->size()<<"\n";
      uint64_t currVert = currQueue->front();
      verts_in_queue.erase(currVert);
      //std::cout<<"HIGH-LOW visiting vtx "<<currVert<<"\n";
      currQueue->pop();
      //if the vertex was previously computed, skip
      if(highs[currVert] != 0 && lows[currVert] != 0) continue;
      bool calculated = true;
      uint64_t low = preorder[currVert];
      uint64_t high = preorder[currVert];
      
      int out_degree = out_degree(g,currVert);
      uint64_t* outs = out_vertices(g,currVert);
      std::unordered_set<uint64_t> nbors_visited;
      //std::cout<<"visiting neighbors\n";
      for(int nborIdx = 0; nborIdx < out_degree; nborIdx++){
        uint64_t local_nbor = outs[nborIdx];
	if(nbors_visited.count(local_nbor) > 0) continue;
	nbors_visited.insert(local_nbor);
        uint64_t global_nbor = 0;
        if(local_nbor < g->n_local) global_nbor = g->local_unmap[local_nbor];
        else global_nbor = g->ghost_unmap[local_nbor-g->n_local];
        //we're looking at a descendant here (do NOT consider edges to parents of currVert)
        if(parents[local_nbor] == g->local_unmap[currVert]){
          if(lows[local_nbor] == 0 || highs[local_nbor] == 0){
            calculated = false;
          } else {
            if(lows[local_nbor] < low) {
              low = lows[local_nbor];
              
            }
            if(highs[local_nbor] > high) high = highs[local_nbor];
          }
        } else if (parents[currVert] != global_nbor) {
          if(preorder[local_nbor] < low) low = preorder[local_nbor];
          if(preorder[local_nbor] > high) high = preorder[local_nbor];
        }
      }
      //std::cout<<"done visiting neighbors\n";
      //calculate the current vertex if possible, if not push it onto the secondary queue
      //if current vertex was calculated, push parent on the secondary frontier
      if(calculated){
	//std::cout<<"inside calculated branch\n";
        highs[currVert] = high;
	//std::cout<<"set highs[currVert] = "<<highs[currVert]<<"\n";
        lows[currVert] = low;
	//std::cout<<"set lows[currVert] = "<<lows[currVert]<<"\n";
	//std::cout<<"parents[currVert] = "<<parents[currVert]<<"\n";
	uint64_t local_parent = get_value(g->map, parents[currVert]);
	//std::cout<<"local_parent = "<<local_parent<<"\n";
        if(local_parent < g->n_local && verts_in_queue.count(local_parent) == 0){
	  //std::cout<<"inside branch inside calc\n";
          otherQueue->push(local_parent);
	  verts_in_queue.insert(local_parent);
        }
      } else {
	//std::cout<<"inside other branch\n";
        if(verts_in_queue.count(currVert) == 0){
	  otherQueue->push(currVert);
	  verts_in_queue.insert(currVert);
	}
      }
      //std::cout<<"done with vertex "<<currVert<<"\n";
    }
    //std::cout<<"communicating preorder labels\n";
    //communicate the highs and lows using the frontier comm function, secondary queue as input
    communicate_preorder_labels(g,*otherQueue,highs,0);
    communicate_preorder_labels(g,*otherQueue,lows,0);
    //std::cout<<"done communicating\n";
    //see if secondary queues are all empty with an MPI_Allreduce, MPI_SUM
    int secondary = otherQueue->size();
    MPI_Allreduce(&secondary, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //std::cout<<"global_done = "<<global_done<<"\n";
    global_done = !global_done;
    
    //switch secondary and primary queues
    std::swap(currQueue,otherQueue);
  }
  //from leaves, compute highs and lows.
  /*uint64_t unfinished = g->n_total;
  uint64_t last_unfinished = g->n_total+1;
  int global_done = 0;
  int local_done = 0;
  while(!global_done){
    while(unfinished < last_unfinished){
      last_unfinished = unfinished;
      unfinished = 0;
      
      for(int parent = 0; parent < g->n_local; parent++){
        bool calculated = true;
        uint64_t low = preorder[parent];
        uint64_t high = preorder[parent];

        int out_degree = out_degree(g, parent);
        uint64_t* outs = out_vertices(g,parent);
        for(int j = 0; j < out_degree; j++){
          int local_nbor = outs[j];
          if(parents[local_nbor] == g->local_unmap[parent]){
            if(lows[local_nbor] == 0 || highs[local_nbor] == 0){
              calculated = false;
            } else {
              if(lows[local_nbor] < low) low = lows[local_nbor];
              if(highs[local_nbor] > high) high = highs[local_nbor];
            }
          } else {
            if(preorder[local_nbor] < low)  low = preorder[local_nbor];
            if(preorder[local_nbor] > high) high = preorder[local_nbor];
          }
        }
        if(calculated){
          highs[parent] = high;
          lows[parent] = low;
        } else {
          unfinished ++;
        }
      }
      
    }
    local_done = last_unfinished == 0;
    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
    //if not done, ghost->owned->ghost exchanges (need to think more about what type of operation needs done with each)
    if(!global_done){
      //break;
      owned_to_ghost_value_comm(g,comm,q,lows);
      owned_to_ghost_value_comm(g,comm,q,highs);
    }
  }*/  
}

bool edge_is_owned(dist_graph_t* g, uint64_t curr_vtx, uint64_t nbor){
  //if the highest GID vertex in the edge is owned, then the edge is owned.
  uint64_t curr_vtx_GID = 0;
  bool curr_is_owned = false;
  uint64_t nbor_GID = 0;
  bool nbor_is_owned = false;
  if(curr_vtx < g->n_local) {
    curr_vtx_GID = g->local_unmap[curr_vtx];
    curr_is_owned = true;
  } else if( curr_vtx != NULL_KEY){
    curr_vtx_GID = g->ghost_unmap[curr_vtx-g->n_local];
  } else {
    return false;
  }

  if(nbor < g->n_local) {
    nbor_GID = g->local_unmap[nbor];
    nbor_is_owned = true;
  } else if (nbor != NULL_KEY){
    nbor_GID = g->ghost_unmap[nbor-g->n_local];
  } else {
    return false;
  }
  
  if((curr_is_owned && curr_vtx_GID > nbor_GID) || (nbor_is_owned && nbor_GID > curr_vtx_GID)){
    /*if((curr_vtx_GID == 24574 || curr_vtx_GID == 0 || curr_vtx_GID == 8208 ) && (nbor_GID == 0 || nbor_GID == 8208 || nbor_GID == 24574)){
      std::cout<<"curr_vtx_GID = "<<curr_vtx_GID<<" nbor_GID = "<<nbor_GID<<" is owned by this process\n";
    }*/
    return true;
  }

  /*if((curr_vtx_GID == 24574 || nbor_GID == 24574) && (curr_vtx_GID == 0 || nbor_GID == 0 || curr_vtx_GID == 8208 || nbor_GID == 8202)){
    std::cout<<"curr_vtx_GID = "<<curr_vtx_GID<<" nbor_GID = "<<nbor_GID<<" is NOT owned by this process\n";
  }*/
  return false;
}

void create_aux_graph(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, dist_graph_t* aux_g, uint64_t* preorder, 
                      uint64_t* lows, uint64_t* highs, uint64_t* n_desc, uint64_t* parents){
  
  uint64_t n_edges = 0;
  std::cout<<"Calculating the number of edges in the aux graph\n";
  uint64_t edges_to_request = 0;
  uint64_t edges_to_send = 0;
  uint64_t* request_edgelist = new uint64_t[g->m_local*2];
  uint32_t* procs_to_send = new uint32_t[g->m_local];
  int32_t* sendcounts = new int32_t[nprocs];
  for(int i = 0; i < nprocs; i++) sendcounts[i] = 0;
  //count self edges for each tree edge (maybe nontree edges are necessary as well, maybe not. We'll see).
  for(uint64_t v = 0; v < g->n_local; v++){
    std::unordered_set<uint64_t> nbors_visited;
    for(uint64_t w_idx = g->out_degree_list[v]; w_idx < g->out_degree_list[v+1]; w_idx++){
      uint64_t w = g->out_edges[w_idx];
      if(nbors_visited.count(w) >0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(parents[v] == w_global || parents[w] == g->local_unmap[v]){
        //add a self edge in srcs and dsts
        if(edge_is_owned(g,v,w)){
          n_edges++;
        }
        //n_dsts++;
      }
    }
  }
  //non self-edges
  for(uint64_t v = 0; v < g->n_local; v++){
    std::unordered_set<uint64_t> nbors_visited;
    for(uint64_t j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; j++){
      uint64_t w = g->out_edges[j];
      if(nbors_visited.count(w) > 0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];

      bool aux_endpoint_1_owned = true;
      bool aux_endpoint_2_owned = true;
      bool will_add_aux = false;
      uint64_t v_parent_lid = get_value(g->map, parents[v]);
      uint64_t w_parent_lid = get_value(g->map, parents[w]);

      if(parents[w] != g->local_unmap[v] && parents[v] != w_global){ //nontree edge
        if(preorder[v] + n_desc[v] <= preorder[w] || preorder[w] + n_desc[w] <= preorder[v]){
          //add{{parents[v], v}, {parents[w], w}} as an edge (w may be ghosted, so parents[w] may be remote)
          will_add_aux = true;
          aux_endpoint_1_owned = edge_is_owned(g,v,v_parent_lid);
          aux_endpoint_2_owned = edge_is_owned(g,w,w_parent_lid);

          //request w,parent[w] from owner of w, if w is not local, need GEI and owner of GEI
          if(w >= g->n_local && w_parent_lid >= g->n_local){
            int w_owner = g->ghost_tasks[w-g->n_local];
            procs_to_send[edges_to_request/2] = w_owner;
            request_edgelist[edges_to_request++] = w_global;
            request_edgelist[edges_to_request++] = parents[w];
            sendcounts[w_owner] += 4; //extra value for owner of GEI
          }
        }
      } else{
        if(parents[w] == g->local_unmap[v]){
          if(preorder[v] != 1 && (lows[w] < preorder[v] || highs[w] >= preorder[v] + n_desc[v])){
            //add{{parents[v],v},{v,w}} as an edge (endpoints guaranteed to exist, so all GEIs will be local)
            will_add_aux = true;
            aux_endpoint_1_owned = edge_is_owned(g,v,v_parent_lid);
            aux_endpoint_2_owned = edge_is_owned(g,v,w);
          }
        } else if (parents[v] == w_global){
          if(preorder[w] != 1 && (lows[v] < preorder[w] || highs[v] >= preorder[w] + n_desc[w])){
            //add {{parents[w], w}, {w, v}} as an edge (w may be ghosted, so parents[w] may be remote)
            will_add_aux = true;
            aux_endpoint_1_owned = edge_is_owned(g,w,w_parent_lid);
            aux_endpoint_2_owned = edge_is_owned(g,v,w);

            //request w, parents[w] from owner of w, if not local, need GEI and owner of GEI
            if(w >= g->n_local && w_parent_lid >= g->n_local){
              int w_owner = g->ghost_tasks[w-g->n_local];
              procs_to_send[edges_to_request/2] = w_owner;
              request_edgelist[edges_to_request++] = w_global;
              request_edgelist[edges_to_request++] = parents[w];
              sendcounts[w_owner] += 4; //extra value for owner of GEI
            }
          }
        }
      }
      //see if the endpoints are local or nah
      if(will_add_aux){
        n_edges += aux_endpoint_1_owned + aux_endpoint_2_owned;
        edges_to_send += !aux_endpoint_1_owned + !aux_endpoint_2_owned;
      }
    }
  }
  
  
  int32_t* recvcounts = new int32_t[nprocs];
  //communicate sendcounts
  MPI_Alltoall(sendcounts, 1, MPI_INT32_T, recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

  int32_t* sdispls = new int32_t[nprocs+1];
  int32_t* rdispls = new int32_t[nprocs+1];
  int32_t* sdispls_cpy = new int32_t[nprocs+1];
  sdispls[0] = 0;
  rdispls[0] = 0;
  sdispls_cpy[0] = 0;
  for(int32_t i = 1; i < nprocs+1; i++){
    sdispls[i] = sdispls[i-1] + sendcounts[i-1];
    rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    sdispls_cpy[i] = sdispls[i];
  }
  int32_t send_total = sdispls[nprocs-1] + sendcounts[nprocs-1];
  int32_t recv_total = rdispls[nprocs-1] + recvcounts[nprocs-1];
  uint64_t* sendbuf = new uint64_t[send_total];
  uint64_t* recvbuf = new uint64_t[recv_total];
  //request edges in request_edgelist, each of which are owned by procs_to_send[i/2]
  //request_edgelist has edges_to_request edges.
  for(int i = 0; i < edges_to_request; i+=2){
    sendbuf[sdispls_cpy[procs_to_send[i/2]]++] = request_edgelist[i];
    sendbuf[sdispls_cpy[procs_to_send[i/2]]++] = request_edgelist[i+1];
    sendbuf[sdispls_cpy[procs_to_send[i/2]]++] = 0;//this will be sent back with the GEI for the requested edge.
    sendbuf[sdispls_cpy[procs_to_send[i/2]]++] = 0;//this will be sent back with the owner of the requested GEI (not nec. the owner of w)
  }
  MPI_Alltoallv(sendbuf, sendcounts, sdispls,MPI_UINT64_T, recvbuf,recvcounts,rdispls,MPI_UINT64_T,MPI_COMM_WORLD);

  for(int i = 0; i < recv_total; i+=4){
    uint64_t vert1_g = recvbuf[i];
    uint64_t vert2_g = recvbuf[i+1];
    uint64_t vert1_l = get_value(g->map, vert1_g);
    uint64_t vert2_l = get_value(g->map, vert2_g);
    bool owned = edge_is_owned(g,vert1_l,vert2_l);
    if(vert1_l < g->n_local){
      for(int j = g->out_degree_list[vert1_l]; j < g->out_degree_list[vert1_l+1]; j++){
        if(g->out_edges[j] == vert2_l){
          recvbuf[i+2] = g->edge_unmap[j];
          //determine who owns this GEI
          if(owned) recvbuf[i+3] = procid;
          else recvbuf[i+3] = g->ghost_tasks[vert2_l-g->n_local];
          break;
        }
      }
    } else {
      for(int j = g->out_degree_list[vert2_l]; j< g->out_degree_list[vert2_l+1]; j++){
        if(g->out_edges[j] == vert1_l){
          recvbuf[i+2] = g->edge_unmap[j];
          //determine who owns this GEI
          if(owned) recvbuf[i+3] = procid;
          else recvbuf[i+3] = g->ghost_tasks[vert1_l-g->n_local];
          break;
        }
      }
    }
    /*if(vert1_g == 8208 && vert2_g == 0){
      std::cout<<"sending back Global Edge Index "<<recvbuf[i+2]<<" for edge between "<<vert1_g<<" and "<<vert2_g<<"\n";
    }*/
  }
  //send back the data
  MPI_Alltoallv(recvbuf,recvcounts,rdispls,MPI_UINT64_T, sendbuf,sendcounts,sdispls,MPI_UINT64_T, MPI_COMM_WORLD);
  
  std::unordered_map<std::pair<uint64_t, uint64_t>,uint64_t,pair_hash> remote_global_edge_indices;
  std::unordered_map<uint64_t, uint64_t> remote_global_edge_owners;

  for(int i = 0; i < nprocs; i++){
    for(int j = sdispls[i]; j < sdispls[i+1]; j+=4){
      uint64_t vert1_g = sendbuf[j];
      uint64_t vert2_g = sendbuf[j+1];
      uint64_t global_edge_index = sendbuf[j+2];
      uint64_t global_edge_owner = sendbuf[j+3];
      remote_global_edge_indices[std::make_pair(vert1_g,vert2_g)] = global_edge_index;
      remote_global_edge_indices[std::make_pair(vert2_g,vert1_g)] = global_edge_index;
      remote_global_edge_owners[global_edge_index] = global_edge_owner;
      //if(global_edge_index == 2 || (vert1_g == 8208 && vert2_g == 0)) std::cout<<"Task "<<procid<<": received global edge "<<global_edge_index<<" from process "<<i<<"\n";

    }
  }

  uint64_t* srcs = (uint64_t*)malloc(n_edges*2*sizeof(uint64_t));//new uint64_t[n_edges*2];
  uint64_t* ghost_edges_to_send = (uint64_t*)malloc(edges_to_send*2*sizeof(uint64_t));
  int* ghost_procs_to_send = (int*) malloc(edges_to_send*sizeof(int));
  int* remote_endpoint_owners= (int*) malloc(edges_to_send*sizeof(int));
  //entries in these arrays take the form {v, w}, where v and w are global vertex identifiers.
  //additionally we ensure v < w.
  for(uint64_t i = 0; i < n_edges*2; i++){
    srcs[i] = 0;
  }
  for(uint64_t i = 0; i < edges_to_send; i++){
    ghost_edges_to_send[i*2] = 0;
    ghost_edges_to_send[i*2+1] = 0;
    ghost_procs_to_send[i] = 0;
    remote_endpoint_owners[i] = 0;
  }
  uint64_t srcIdx = 0;
  uint64_t ghost_edge_idx = 0;
  uint64_t max_edge_GID = 0;
  std::cout<<"Creating srcs and dsts arrays, "<<n_edges<<" edges to be created\n";
  //self edges
  for(uint64_t v = 0; v < g->n_local; v++){
    std::unordered_set<uint64_t> nbors_visited;
    for(uint64_t w_idx = g->out_degree_list[v]; w_idx < g->out_degree_list[v+1]; w_idx++){
      uint64_t w = g->out_edges[w_idx];
      if(nbors_visited.count(w) >0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(parents[v] == w_global || parents[w] == g->local_unmap[v]){
        //add a self edge
        if(edge_is_owned(g,v,w)/*edge {v,w} is owned*/){
          srcs[srcIdx++] = g->edge_unmap[w_idx];
          srcs[srcIdx++] = g->edge_unmap[w_idx];
        }
      }
    }
  }
  

  //non-self edges
  for(uint64_t v = 0; v < g->n_local; v++){
    std::unordered_set<uint64_t> nbors_visited;
    for(uint64_t j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; j++){
      uint64_t w = g->out_edges[j];//w is local, not preorder
      if(nbors_visited.count(w) > 0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w < g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];

      bool will_add_aux = false;
      uint64_t v_parent_lid = get_value(g->map, parents[v]);
      uint64_t w_parent_lid = get_value(g->map, parents[w]); 
      uint64_t aux_endpoint_1_gei = -1;
      uint64_t aux_endpoint_2_gei = -1;
      int aux_endpoint_1_owner = -1;
      int aux_endpoint_2_owner = -1;


      if(parents[w] != g->local_unmap[v] && parents[v] != w_global){ //nontree edge
        if(preorder[v] + n_desc[v] <= preorder[w] || preorder[w] + n_desc[w] <= preorder[v]){
          //add {{parents[v], v}, {parents[w], w}} (undirected edges)
          will_add_aux = true;
          //find GEI of {parents[v],v} (looking up v will always have the edge, as v is owned locally)
          for(int e = g->out_degree_list[v]; e < g->out_degree_list[v+1]; e++){
            if(g->out_edges[e] == get_value(g->map,parents[v])){
              aux_endpoint_1_gei = g->edge_unmap[e];
              break;
            }
          }
          //find owner of {parents[v],v} (if parents[v] is ghost, either this process or parents[v] owner owns this aux endpoint)
          if(edge_is_owned(g,v,v_parent_lid)) aux_endpoint_1_owner = procid;
          else aux_endpoint_1_owner = g->ghost_tasks[v_parent_lid-g->n_local];

          //find GEI of {parents[w],w} (w could be local or ghosted, and parents[w] local, ghosted, or remote)
          if(w_parent_lid < g->n_local){
            for(int e = g->out_degree_list[w_parent_lid]; e < g->out_degree_list[w_parent_lid+1]; e++){
              if(g->out_edges[e] == w){
                aux_endpoint_2_gei = g->edge_unmap[e];
                break;
              }
            }
            //aux_endpoint_2_owner is either this proc, or owner[w]
            if(edge_is_owned(g,w,w_parent_lid)) aux_endpoint_2_owner = procid;
            else aux_endpoint_2_owner = g->ghost_tasks[w-g->n_local];

          } else if(w < g->n_local && w_parent_lid != NULL_KEY){
            for(int e = g->out_degree_list[w]; e < g->out_degree_list[w+1]; e++){
              if(g->out_edges[e] == w_parent_lid){
                aux_endpoint_2_gei = g->edge_unmap[e];
                break;
              }
            }
            //aux_endpoint_2_owner is either this proc, or owner[parent[w]]
            if(edge_is_owned(g,w,w_parent_lid)) aux_endpoint_2_owner = procid;
            else aux_endpoint_2_owner = g->ghost_tasks[w_parent_lid-g->n_local];
          } else {
            //owner is given by remote_global_edge_owners, aux_endpoint_2_gei is given by remote_global_edge_indices
            std::pair<uint64_t, uint64_t> edge = std::make_pair(parents[w],w_global);
            aux_endpoint_2_gei = remote_global_edge_indices.at(edge);
            aux_endpoint_2_owner = remote_global_edge_owners.at(aux_endpoint_2_gei);
          }

        }
      } else{
        if(parents[w] == g->local_unmap[v]){
          if(preorder[v] != 1 && (lows[w] < preorder[v] || highs[w] >= preorder[v] + n_desc[v])){
            //add {{parents[v], v} , {v, w}} as an edge
            will_add_aux = true;
            //find GEI of {parents[v],v} (looking up v will always have the edge, as v is owned locally)
            for(int e = g->out_degree_list[v]; e < g->out_degree_list[v+1]; e++){
              if(g->out_edges[e] == v_parent_lid){
                aux_endpoint_1_gei = g->edge_unmap[e];
                break;
              }
            }
            //the aux endpoint is either owned by this proc or the owner of parents[v]
            //(if the edge isn't owned here, parents[v] must be a ghost)
            if(edge_is_owned(g,v,v_parent_lid)) aux_endpoint_1_owner = procid;
            else aux_endpoint_1_owner = g->ghost_tasks[v_parent_lid-g->n_local];
            
            //find GEI of {w,v} (looking up v will always have the edge, v is owned locally)
            for(int e = g->out_degree_list[v]; e < g->out_degree_list[v+1]; e++){
              if(g->out_edges[e] == w){
                aux_endpoint_2_gei = g->edge_unmap[e];
                break;
              }
            }
            //edge is either owned by this process, or the owner of w.
            if(edge_is_owned(g,v,w)) aux_endpoint_2_owner = procid;
            else aux_endpoint_2_owner = g->ghost_tasks[w-g->n_local];
          }
        } else if (parents[v] == w_global){
          if(preorder[w] != 1 && (lows[v] < preorder[w] || highs[v] >= preorder[w] + n_desc[w])){
            //add {{parents[w], w}, {w, v}} as an edge
            will_add_aux = true;

            //find GEI for {parents[w], w} (w may be local or ghosted, parents[w] local, ghosted, or remote)
            if(w_parent_lid < g->n_local){ //parents[w] is local, w ghosted or local
              for(int e = g->out_degree_list[w_parent_lid]; e < g->out_degree_list[w_parent_lid+1]; e++){
                if(g->out_edges[e] == w){
                  aux_endpoint_1_gei = g->edge_unmap[e];
                  break;
                }
              }
              //aux_endpoint_1_owner is either this proc or owner[w]
              if(edge_is_owned(g,w,w_parent_lid)) aux_endpoint_1_owner = procid;
              else aux_endpoint_1_owner = g->ghost_tasks[w-g->n_local];

            } else if(w < g->n_local && w_parent_lid != NULL_KEY){//w is local, parents[w] is ghosted
              for(int e = g->out_degree_list[w]; e < g->out_degree_list[w+1]; e++){
                if(g->out_edges[e] == w_parent_lid){
                  aux_endpoint_1_gei = g->edge_unmap[e];
                  break;
                }
              }
              //edge is either owned by this process or owner of parents[w]
              if(edge_is_owned(g,w,w_parent_lid)) aux_endpoint_1_owner = procid;
              else aux_endpoint_1_owner = g->ghost_tasks[w_parent_lid-g->n_local];

            } else {//both w and parents[w] are ghosted, or w is ghosted and parents[w] is remote
              std::pair<uint64_t, uint64_t> edge = std::make_pair(parents[w],w_global);
              aux_endpoint_1_gei = remote_global_edge_indices.at(edge);
              aux_endpoint_1_owner = remote_global_edge_owners.at(aux_endpoint_1_gei);
            }

            //find GEI for {w,v} (looking up v will always have the edge, v is owned locally)
            for(int e = g->out_degree_list[v]; e < g->out_degree_list[v+1]; e++){
              if(g->out_edges[e] == w){
                aux_endpoint_2_gei = g->edge_unmap[e];
                break;
              }
            }
            //aux_endpoint_2_owner is either this proc or owner[w]
            if(edge_is_owned(g,v,w)) aux_endpoint_2_owner = procid;
            else aux_endpoint_2_owner = g->ghost_tasks[w-g->n_local];
          }
        }
      }

      if(will_add_aux){
        //an aux edge needs to be added from aux_endpoint_1_gei to aux_endpoint_2_gei.
        //aux_endpoint_1_gei is owned by aux_endpoint_1_owner, aux_endpoint_2_gei is owned by aux_endpoint_2_owner
        if(aux_endpoint_1_owner == procid){
          srcs[srcIdx++] = aux_endpoint_1_gei;
          srcs[srcIdx++] = aux_endpoint_2_gei;
        } else {
          ghost_procs_to_send[ghost_edge_idx/2] = aux_endpoint_1_owner;
          remote_endpoint_owners[ghost_edge_idx/2] = aux_endpoint_2_owner;
          ghost_edges_to_send[ghost_edge_idx++] = aux_endpoint_1_gei;
          ghost_edges_to_send[ghost_edge_idx++] = aux_endpoint_2_gei;
        }
        
        if(aux_endpoint_2_owner == procid){
          srcs[srcIdx++] = aux_endpoint_2_gei;
          srcs[srcIdx++] = aux_endpoint_1_gei;
        } else {
          ghost_procs_to_send[ghost_edge_idx/2] = aux_endpoint_2_owner;
          remote_endpoint_owners[ghost_edge_idx/2] = aux_endpoint_1_owner;
          ghost_edges_to_send[ghost_edge_idx++] = aux_endpoint_2_gei;
          ghost_edges_to_send[ghost_edge_idx++] = aux_endpoint_1_gei;
        }
      }
    }
  }
  std::cout<<"Finished populating srcs and dsts arrays:\n\t";
  
  //ghost_edges_to_send are the edges that this process knows, but other processes need in order to construct a complete ghost layer.

  int32_t* ghost_sendcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  int32_t* ghost_recvcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  int32_t* ghost_sdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  int32_t* ghost_rdispls = (int32_t*)malloc((nprocs+1)*sizeof(int32_t));
  int32_t* ghost_sdispls_cpy = (int32_t*)malloc(nprocs*sizeof(int32_t));

  int32_t* remow_sendcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  int32_t* remow_recvcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  int32_t* remow_sdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  int32_t* remow_rdispls = (int32_t*)malloc((nprocs+1)*sizeof(int32_t));
  int32_t* remow_sdispls_cpy = (int32_t*)malloc(nprocs*sizeof(int32_t));




  for(int i = 0; i < nprocs; i++){
    ghost_sendcounts[i] = 0;
    ghost_recvcounts[i] = 0;
    ghost_sdispls[i] = 0;
    ghost_rdispls[i] = 0;
    ghost_sdispls_cpy[i] = 0;

    remow_sendcounts[i] = 0;
    remow_recvcounts[i] = 0;
    remow_sdispls[i] = 0;
    remow_rdispls[i] = 0;
    remow_sdispls_cpy[i] = 0;
  }

  for(int i = 0; i < edges_to_send; i++){
    ghost_sendcounts[ghost_procs_to_send[i]]+=2;
    remow_sendcounts[ghost_procs_to_send[i]]++;
  }
  
  MPI_Alltoall(ghost_sendcounts, 1, MPI_INT32_T, ghost_recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);
  MPI_Alltoall(remow_sendcounts, 1, MPI_INT32_T, remow_recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

  ghost_sdispls[0] = 0;
  ghost_sdispls_cpy[0] = 0;
  ghost_rdispls[0] = 0;
  remow_sdispls[0] = 0;
  remow_sdispls_cpy[0] = 0;
  remow_rdispls[0] = 0;
  for(int32_t i = 1; i < nprocs; i++){
    ghost_sdispls[i] = ghost_sdispls[i-1] + ghost_sendcounts[i-1];
    ghost_rdispls[i] = ghost_rdispls[i-1] + ghost_recvcounts[i-1];
    ghost_sdispls_cpy[i] = ghost_sdispls[i];

    remow_sdispls[i] = remow_sdispls[i-1] + remow_sendcounts[i-1];
    remow_rdispls[i] = remow_rdispls[i-1] + remow_recvcounts[i-1];
    remow_sdispls_cpy[i] = remow_sdispls[i];
  }

  int32_t ghost_send_total = ghost_sdispls[nprocs-1] + ghost_sendcounts[nprocs-1];
  int32_t ghost_recv_total = ghost_rdispls[nprocs-1] + ghost_recvcounts[nprocs-1];

  int32_t remow_send_total = remow_sdispls[nprocs-1] + remow_sendcounts[nprocs-1];
  int32_t remow_recv_total = remow_rdispls[nprocs-1] + remow_recvcounts[nprocs-1];
  ghost_rdispls[nprocs] = ghost_recv_total;
  //realloc more room on the end of the edgelist, so we can simply tack on the extra edges at the end of the edge list.
  srcs = (uint64_t*) realloc(srcs,(n_edges*2+ghost_recv_total)*sizeof(uint64_t));
  uint64_t* ghost_sendbuf = (uint64_t*) malloc((uint64_t)ghost_send_total*sizeof(uint64_t));
  int * remote_endpoint_owner_sendbuf = (int*) malloc((uint64_t)remow_send_total*sizeof(int));
  int * remote_endpoint_owner_recvbuf = (int*) malloc((uint64_t)remow_recv_total*sizeof(int));

  for(uint64_t i = 0; i < ghost_send_total; i+=2){
    uint64_t vert1 = ghost_edges_to_send[i];
    uint64_t vert2 = ghost_edges_to_send[i+1];
    //if(vert1 == 497433 || vert2 == 497433) std::cout<<"Sending edge "<<vert1<<" "<<vert2<<" to process "<<ghost_procs_to_send[i/2]<<"\n";
    ghost_sendbuf[ghost_sdispls_cpy[ghost_procs_to_send[i/2]]++] = vert1;
    ghost_sendbuf[ghost_sdispls_cpy[ghost_procs_to_send[i/2]]++] = vert2;
  }
  for(uint64_t i = 0; i < remow_send_total; i++){
    remote_endpoint_owner_sendbuf[remow_sdispls_cpy[ghost_procs_to_send[i]]++] = remote_endpoint_owners[i];
  }
  MPI_Alltoallv(remote_endpoint_owner_sendbuf, remow_sendcounts, remow_sdispls, MPI_INT,
                remote_endpoint_owner_recvbuf, remow_recvcounts, remow_rdispls, MPI_INT, MPI_COMM_WORLD);
  MPI_Alltoallv(ghost_sendbuf, ghost_sendcounts, ghost_sdispls, MPI_UINT64_T,
                srcs+n_edges*2, ghost_recvcounts, ghost_rdispls, MPI_UINT64_T, MPI_COMM_WORLD);
  for(int i = 0; i < nprocs; i++){
    for(int j = ghost_rdispls[i]; j < ghost_rdispls[i+1]; j+=2){
      uint64_t idx = n_edges*2+j+1;
      /*if(srcs[idx-1] == 497433) std::cout<<"****** received aux vertex 497433 as a source, but it's not owned\n";
      if(srcs[idx] == 497433) std::cout<<"******* received aux vertex 497433 as a dest\n";*/
      if(remote_global_edge_owners.count(srcs[idx]) == 0){
        remote_global_edge_owners[srcs[idx]] = remote_endpoint_owner_recvbuf[j/2]; 
      }
    }
  }
  
  n_edges += ghost_recv_total/2;
  /*for(int i = 0; i < srcIdx; i++){
    std::cout<<srcs[i]<<" ";
  }
  std::cout<<"\n";*/
  aux_g->map = (struct fast_map*)malloc(sizeof(struct fast_map));
  if((g->m_local + g->n_local)*2 < g->n) init_map_nohash(aux_g->map, g->n);
  else init_map(aux_g->map, (g->m_local+g->n_local)*2);
  aux_g->local_unmap = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  
  uint64_t curr_lid = 0;
  aux_g->m_local = n_edges;
  //std::cout<<"n_edges = "<<n_edges<<" srcIdx = "<<srcIdx<<"\n";
  aux_g->out_edges = (uint64_t*)malloc(aux_g->m_local*sizeof(uint64_t));
  /*for(uint64_t i = 0; i < aux_g->m_local; i++){
    std::cout<<"accessing aux_g->out_edges["<<i<<"]\n";
    aux_g->out_edges[i] = 0;
  }*/
  ///std::cout<<"done with edge size test\n";
  uint64_t* temp_counts =  (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
//#pragma omp parallel for
  for(uint64_t i = 0; i < g->m_local; i++) temp_counts[i] = 0;

  //run through global IDs in the src position, map to global IDs, add to local_unmap, and update temp_counts
  for(uint64_t i = 0; i < aux_g->m_local*2; i+=2){
    if(get_value(aux_g->map, srcs[i]) == NULL_KEY){
      aux_g->local_unmap[curr_lid] = srcs[i];
      temp_counts[curr_lid]++;
      set_value(aux_g->map, srcs[i], curr_lid);
      curr_lid++;
    } else {
      temp_counts[get_value(aux_g->map,srcs[i])]++; 
    }
  }
  aux_g->n_local = curr_lid;
  /*for(uint64_t i = 0; i < aux_g->n_local; i++){
    std::cout<<"local "<<i<<" is global "<<aux_g->local_unmap[i]<<"\n";
  }*/
  aux_g->out_degree_list = (uint64_t*)malloc((aux_g->n_local+1)*sizeof(uint64_t));

//#pragma omp parallel for
  for(uint64_t i = 0; i< aux_g->n_local+1; i++) aux_g->out_degree_list[i] = 0;

  for(uint64_t i = 0; i < aux_g->n_local; i++){
    //std::cout<<"vertex "<<aux_g->local_unmap[i] <<" has degree "<<temp_counts[i]<<"\n";
    aux_g->out_degree_list[i+1] = aux_g->out_degree_list[i] + temp_counts[i];
  }

  memcpy(temp_counts, aux_g->out_degree_list, aux_g->n_local*sizeof(uint64_t));
  assert(aux_g->out_degree_list[aux_g->n_local] == aux_g->m_local); 
  for(uint64_t i = 0; i < aux_g->m_local*2; i+=2){
    uint64_t global_src = srcs[i];
    uint64_t global_dest = srcs[i+1];
    if(global_src == 438 && global_dest == 2) std::cout<<"Task "<<procid<<": 438 neighbors 2***********************************\n";
    aux_g->out_edges[temp_counts[get_value(aux_g->map,global_src)]++] = global_dest;
  }


  aux_g->ghost_tasks = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  aux_g->ghost_unmap = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  for(uint64_t i = 0; i < aux_g->m_local; i++){
    uint64_t global_dest = aux_g->out_edges[i];
    uint64_t val = get_value(aux_g->map, global_dest);
    if(val == NULL_KEY){
      //std::cout<<"global_dest = "<<global_dest<<" and is a ghost\n";
      set_value(aux_g->map, global_dest, curr_lid);
      aux_g->ghost_unmap[curr_lid-aux_g->n_local] = global_dest;
      
      uint64_t local_edge_index = get_value(g->edge_map, global_dest);
      if(local_edge_index != NULL_KEY){
        
        aux_g->ghost_tasks[curr_lid-aux_g->n_local] = g->ghost_tasks[g->out_edges[local_edge_index]-g->n_local];
      } else {
        aux_g->ghost_tasks[curr_lid-aux_g->n_local] = remote_global_edge_owners.at(global_dest);
      }
      /*if(global_dest == 2) std::cout<<"Task "<<procid<<" thinks global aux vertex "<<global_dest<<" is owned by process "<<aux_g->ghost_tasks[curr_lid-aux_g->n_local]<<"\n";*/
      aux_g->out_edges[i] = curr_lid++;
    } else {
      aux_g->out_edges[i] = val;
    }
  }
  /*for(uint64_t i = 0; i < aux_g->m_local*2; i+=2){
    uint64_t global_src = srcs[i];
    uint64_t global_dest = srcs[i+1];
    uint64_t local_src = get_value(aux_g->map, global_src);
    //if global_dest is not in aux_g->map, add it and place the local ID in its place in out_edges
    if(get_value(aux_g->map, global_dest) == NULL_KEY){
      aux_g->ghost_unmap[curr_lid - aux_g->n_local] = global_dest;
      aux_g->out_edges[temp_counts[local_src]++] = curr_lid;
      //set the ghost_tasks, 
      //1. lookup which local edge corresponds to the destination (could be entirely remote, though)
      if(get_value(g->edge_map, global_dest) != NULL_KEY){
        uint64_t local_edge_index = get_value(g->edge_map, global_dest);
        //the owner of the destination vertex of this edge should be ghosted (source vertices are never ghosted)
        aux_g->ghost_tasks[curr_lid] = g->ghost_tasks[g->out_edges[local_edge_index]-g->n_local];
      } else {
        //2. if there is no corresponding local edge, lookup the remote owner of this edge in the map.
        aux_g->ghost_tasks[curr_lid] = remote_global_edge_owners.at(global_dest);
      }
      set_value(aux_g->map, global_dest, curr_lid++);
    } else {
      //ghost_tasks, ghost_unmap, and map values already set, just translate global->local and place the local value in out_edges
      aux_g->out_edges[temp_counts[local_src]++] = get_value(aux_g->map, global_dest);
    }
  }*/
  aux_g->n_ghost = curr_lid - aux_g->n_local;
  aux_g->n_total = curr_lid;
  /*for(uint64_t i = 0; i < aux_g->n_ghost; i++){
    if(aux_g->ghost_unmap[i] == 4092) std::cout<<"Task "<<procid<<" has vertex "<<aux_g->ghost_unmap[i]<<" in ghost_unmap\n";
  }*/
  for(uint64_t i = 0; i < aux_g->n_local; i++){
    if(aux_g->local_unmap[i] == 438){
      for(int j = aux_g->out_degree_list[i]; j < aux_g->out_degree_list[i+1]; j++){
          //std::cout<<"****Global vertex "<<aux_g->local_unmap[i]<<", local "<<i<<", has neighbor "<<aux_g->out_edges[j]<<"\n";
      }
    }
  }
  /*for(uint64_t i = 0; i < aux_g->n_ghost; i++){
    std::cout<<"ghost vertex "<<i+aux_g->n_local<<" has global ID "<<aux_g->ghost_unmap[i]<<" and is owned by process "<<aux_g->ghost_tasks[i]<<"\n";
  }*/
  //communicate number of local, owned verts:
  MPI_Allreduce(&aux_g->n_local,&aux_g->n,1,MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  //communicate number of local edges, add together
  MPI_Allreduce(&aux_g->m_local,&aux_g->m,1,MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  std::cout<<"aux_g->n = "<<aux_g->n<<" aux_g->m = "<<aux_g->m<<"\n";
  /*graph_gen_data_t* ggi = new graph_gen_data_t;
  //need to calculate ggi->n, ggi->n_local, ggi->m, ggi->m_local_edges
  ggi->n_local = 0;
  for(int i =0; i < g->n_local; i++){
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      uint64_t nbor = g->out_edges[j];
      if(nbor < g->n_local){
        //for local edges, only count one direction, as reverse edges have the same ID
        if(g->local_unmap[i] < g->local_unmap[nbor]){
          ggi->n_local++;
        }
      } else {
        ggi->n_local++;
      }
    }
  }
  
  MPI_Allreduce(&max_edge_GID,&ggi->n,1,MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  uint64_t m_local = srcIdx/2;
  MPI_Allreduce(&m_local,&ggi->m,1,MPI_UINT64_T,MPI_SUM, MPI_COMM_WORLD);
  ggi->m_local_edges = m_local;
  ggi->gen_edges = srcs;
  ggi->global_edge_indices = NULL;
  
  std::cout<<"Task "<<procid<<": n = "<<ggi->n<<" n_local = "<<ggi->n_local<<" m_local_edges = "<<ggi->m_local_edges<<"\n";

  create_graph(ggi, aux_g);
  relabel_edges(aux_g);
  
  //fix up the ghost_tasks
  for(int i = 0; i < aux_g->n_ghost; i++){
    if(remote_global_edge_owners.find(aux_g->ghost_unmap[i]) == remote_global_edge_owners.end())
      std::cout<<"fixing ghost tasks for ghost vertex "<<aux_g->ghost_unmap[i]<<", but it isn't in the remote owners map\n";
      
    aux_g->ghost_tasks[i] = remote_global_edge_owners.at(aux_g->ghost_unmap[i]);
  } */
}



void finish_edge_labeling(dist_graph_t* g,uint64_t* g_srcs, dist_graph_t* aux_g, uint64_t* preorder, uint64_t* parents,
                          uint64_t * aux_labels, uint64_t* final_labels){
  //update final_labels (vertex labels) with the edge labels from aux_labels
  for(uint64_t i = 0; i < g->n_local; i++){
    for(uint64_t j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      uint64_t vert1_lid = i;
      uint64_t vert2_lid = g->out_edges[j];
      uint64_t edge_gid = g->edge_unmap[j];
      uint64_t edge_lid = get_value(aux_g->map, edge_gid);
      if(edge_lid != NULL_KEY){
        if(final_labels[vert1_lid] > aux_labels[edge_lid]){
          final_labels[vert1_lid] = aux_labels[edge_lid];
        }
        if(final_labels[vert2_lid] > aux_labels[edge_lid]){
          final_labels[vert2_lid] = aux_labels[edge_lid];
        }
      }
    }
  }

  int32_t* sendcounts = new int32_t[nprocs];
  int32_t* recvcounts = new int32_t[nprocs];
  for(int i = 0; i < nprocs; i++)sendcounts[i] = 0;
  for(int i = 0; i < g->n_ghost; i++){
    sendcounts[g->ghost_tasks[i]] += 2;
  }
  MPI_Alltoall(sendcounts, 1, MPI_INT32_T, recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

  int32_t* sdispls = new int32_t[nprocs];
  int32_t* rdispls = new int32_t[nprocs];
  int32_t* sdispls_cpy = new int32_t[nprocs];

  sdispls[0] = 0;
  rdispls[0] = 0;
  sdispls_cpy[0] = 0;
  for(int32_t i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcounts[i-1];
    rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    sdispls_cpy[i] = sdispls[i];
  }
  
  int32_t send_total = sdispls[nprocs-1] + sendcounts[nprocs-1];
  int32_t recv_total = rdispls[nprocs-1] + recvcounts[nprocs-1];

  uint64_t* sendbuf = new uint64_t[send_total];
  uint64_t* recvbuf = new uint64_t[recv_total];

  for(int i = 0; i < g->n_ghost; i++){
    sendbuf[sdispls_cpy[g->ghost_tasks[i]]++] = g->ghost_unmap[i];
    sendbuf[sdispls_cpy[g->ghost_tasks[i]]++] = 0;
  }
  //do a boundary exchange
  MPI_Alltoallv(sendbuf,sendcounts, sdispls, MPI_UINT64_T, recvbuf, recvcounts, rdispls, MPI_UINT64_T, MPI_COMM_WORLD);

  for(uint64_t i = 0; i< recv_total; i+=2){
    uint64_t lid = get_value(g->map, recvbuf[i]);
    recvbuf[i+1] = final_labels[lid];
  }

  MPI_Alltoallv(recvbuf, recvcounts, rdispls, MPI_UINT64_T, sendbuf, sendcounts, sdispls, MPI_UINT64_T, MPI_COMM_WORLD);

  for(int i = 0; i < send_total; i+=2){
    uint64_t ghost_lid = get_value(g->map, sendbuf[i]);
    final_labels[ghost_lid] = sendbuf[i+1];
  }
  //done
}

extern "C" int bicc_dist(dist_graph_t* g,mpi_data_t* comm, queue_data_t* q)
{

  double elt = 0.0, elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

  uint64_t* srcs = new uint64_t[g->m_local];
  for(int i = 0; i < g->n_local; i++){
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      srcs[j] = i;
    }
  }
  /*if(get_value(g->edge_map,2) != NULL_KEY){
    std::cout<<"*******Global edge index 2 corresponds to global endpoints "<<g->local_unmap[srcs[get_value(g->edge_map,2)]];
    if(g->out_edges[get_value(g->edge_map,2)] < g->n_local) std::cout<<" "<<g->local_unmap[g->out_edges[get_value(g->edge_map,2)]]<<"\n";
    else std::cout<<" (g)"<<g->ghost_unmap[g->out_edges[get_value(g->edge_map,2)]-g->n_local]<<"\n";
    std::cout<<"global vertex 8208 neighbors: ";
    for(int i = g->out_degree_list[get_value(g->map, 8208)]; i < g->out_degree_list[get_value(g->map,8208)+1]; i++){
      if(g->out_edges[i] >= g->n_local && g->ghost_unmap[g->out_edges[i]-g->n_local] == 24574) 
        std::cout<<g->ghost_unmap[g->out_edges[i]-g->n_local]<<"g, Global edge index is "<<g->edge_unmap[i]<<"\n";
    }
  }*/
  

  /*if(get_value(g->edge_map,365956) != NULL_KEY){
    std::cout<<"*******Global edge index 365956 corresponds to global endpoints "<<g->local_unmap[srcs[get_value(g->edge_map,365956)]];
    if(g->out_edges[get_value(g->edge_map,365956)] < g->n_local) std::cout<<" "<<g->local_unmap[g->out_edges[get_value(g->edge_map,365956)]]<<"\n";
    else std::cout<<" (g)"<<g->ghost_unmap[g->out_edges[get_value(g->edge_map,365956)]-g->n_local]<<"\n";
  }
  if(get_value(g->edge_map,440888) != NULL_KEY){
    std::cout<<"*******Global edge index 440888 corresponds to global endpoints "<<g->local_unmap[srcs[get_value(g->edge_map,440888)]];
    if(g->out_edges[get_value(g->edge_map,440888)] < g->n_local) std::cout<<" "<<g->local_unmap[g->out_edges[get_value(g->edge_map,440888)]]<<"\n";
    else std::cout<<" (g)"<<g->ghost_unmap[g->out_edges[get_value(g->edge_map,440888)]-g->n_local]<<"\n";
  }
  if(get_value(g->map, 48) != NULL_KEY){
    uint64_t lid = get_value(g->map, 48);
    for(uint64_t i = g->out_degree_list[lid]; i < g->out_degree_list[lid+1]; i++){
      uint64_t nbor_lid = g->out_edges[i];
      uint64_t nbor_gid = 0;
      bool ghost = false;
      if(nbor_lid < g->n_local) nbor_gid = g->local_unmap[nbor_lid];
      else {
        ghost = true;
        nbor_gid = g->ghost_unmap[nbor_lid-g->n_local];
      }
      if(nbor_gid == 30716 || nbor_gid == 15551){
        std::cout<<"vertex 48 neighbors "<<nbor_gid;
        if(ghost) std::cout<<"g";
        std::cout<<"\n";
      }
    }
  } else {
    uint64_t lid = get_value(g->map, 30716);
    for(uint64_t i = g->out_degree_list[lid]; i < g->out_degree_list[lid+1]; i++){
      uint64_t nbor_lid = g->out_edges[i];
      uint64_t nbor_gid = 0;
      bool ghost = false;
      if(nbor_lid < g->n_local) nbor_gid = g->local_unmap[nbor_lid];
      else {
        ghost = true;
        nbor_gid = g->ghost_unmap[nbor_lid-g->n_local];
      }
      if(nbor_gid == 48 || nbor_gid == 15507){
        std::cout<<"vertex 30716 neighbors "<<nbor_gid;
        if(ghost) std::cout<<"g";
        std::cout<<"\n";
      }
    }
  }*/
  /*if(get_value(g->edge_map,5459) != NULL_KEY){
    std::cout<<"*******Global edge index 5459 corresponds to global endpoints "<<g->local_unmap[srcs[get_value(g->edge_map,5459)]];
    if(g->out_edges[get_value(g->edge_map,5459)] < g->n_local) std::cout<<" "<<g->local_unmap[g->out_edges[get_value(g->edge_map,5459)]]<<"\n";
    else std::cout<<" (g)"<<g->ghost_unmap[g->out_edges[get_value(g->edge_map,5459)]-g->n_local]<<"\n";
  }
  if(get_value(g->edge_map,273436) != NULL_KEY){
    std::cout<<"*******Global edge index 273436 corresponds to global endpoints "<<g->local_unmap[srcs[get_value(g->edge_map,273436)]];
    if(g->out_edges[get_value(g->edge_map,273436)] < g->n_local) std::cout<<" "<<g->local_unmap[g->out_edges[get_value(g->edge_map,273436)]]<<"\n";
    else std::cout<<" (g)"<<g->ghost_unmap[g->out_edges[get_value(g->edge_map,273436)]-g->n_local]<<"\n";
  }*/
  /*if(get_value(g->edge_map,97345) != NULL_KEY){
    std::cout<<"global edge index 97345 corresponds to local edge "<<get_value(g->edge_map,97345)<<", which has endpoints "
             <<srcs[get_value(g->edge_map,97345)]<<", "<<g->out_edges[get_value(g->edge_map,97345)]<<"\n";
  } else {
    std::cout<<"global edge index 97345 is remote\n";
    for(int i = 0; i < g->n_ghost; i++){
      if(g->ghost_unmap[i] == 4826 || g->ghost_unmap[i] == 10600){
        std::cout<<"vertex "<<g->ghost_unmap[i]<<" is a ghost\n";
      }
    }
  }
  while(true);*/
  /*for(uint64_t i = 0; i < g->m; i++){
    if(get_value(g->edge_map,i) != NULL_KEY){
      uint64_t local_edge_index = get_value(g->edge_map, i);
      std::cout<<"Task "<<procid<<": Global edge ID "<<i<<" associated with edge "<<g->local_unmap[srcs[local_edge_index]];
      if(g->out_edges[get_value(g->edge_map, i)] < g->n_local) std::cout<<" "<<g->local_unmap[g->out_edges[local_edge_index]];
      else std::cout<<" "<<g->ghost_unmap[g->out_edges[local_edge_index]-g->n_local];
      std::cout<<"\n";
    } else {
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }*/
  
  //while(true) MPI_Barrier(MPI_COMM_WORLD);
  /*for(int i = 0; i < g->n_local; i++){
    int out_degree = out_degree(g, i);
    uint64_t* outs = out_vertices(g, i);
    printf("Rank %d: %d's neighbors:\n",procid,g->local_unmap[i]);
    for(int j = 0; j < out_degree; j++){
      if(outs[j] < g->n_local){
        printf("\t%d\n",g->local_unmap[outs[j]]);
      } else {
        printf("\t%d\n",g->ghost_unmap[outs[j] - g->n_local]);
      }
    }
  }*/
  uint64_t* parents = new uint64_t[g->n_total];
  uint64_t* levels = new uint64_t[g->n_total];
  /*uint64_t maxDegree = -1;
  uint64_t maxDegreeVert = -1;
  for(int i=0; i < g->n_local; i++){
    int degree = out_degree(g,i);
    if(degree > maxDegree){
      maxDegree = degree;
      maxDegreeVert = i;
    }
  }*/
  bicc_bfs(g, comm, parents, levels, g->max_degree_vert);
 
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  /*for(int i = 0; i < g->n_local; i++){
    int curr_global = g->local_unmap[i];
    printf("vertex %d, parent: %d, level: %d, is_leaf: %d\n",curr_global, parents[i], levels[i],is_leaf[i]);
  }
  for(int i = 0; i < g->n_ghost; i++){
    int curr = g->n_local + i;
    printf("vertex %d, parent: %d, level: %d, is_leaf: %d\n",g->ghost_unmap[i], parents[curr], levels[curr],is_leaf[curr]);
  }*/
  
  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Calculating descendants\n");
  }

  
  
  //1. calculate descendants for each owned vertex, counting itself as its own descendant
  uint64_t* n_desc = new uint64_t[g->n_total];
  calculate_descendants(g,comm,q,parents,n_desc);
  //std::cout<<"Finished calculating descendants\n";
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("calculating preorder\n");
  }
  /*for(int i = 0; i < g->n_local; i++){
    printf("Rank %d: Vertex %d has %d descendants\n",procid,g->local_unmap[i],n_desc[i]);
  }
  for(int i = g->n_local; i < g->n_total; i++){
    printf("Rank %d: Vertex %d has %d descendants\n",procid,g->ghost_unmap[i-g->n_local],n_desc[i]);  
  }*/
  //2. calculate preorder labels for each owned vertex (using the number of descendants)
  //delete [] is_leaf;
  /*uint64_t* preorder_recursive = new uint64_t[g->n_total];
  for(int i = 0; i < g->n_total; i++) preorder_recursive[i] = NULL_KEY;
  uint64_t pidx = 1;
  preorder_label_recursive(g,parents,preorder_recursive, 3950,pidx);*/

  uint64_t* preorder = new uint64_t[g->n_total];
  for(int i = 0; i < g->n_total; i++) preorder[i] = 0;
  calculate_preorder(g,comm,q,levels,parents,n_desc,preorder); 
  //std::cout<<"Finished calculating preorder labels\n";
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Calculating high low\n");
  }
  /*for(int i = 0; i < g->n_total; i++){
    std::cout<<"vertex "<<i<<" has preorder label "<<preorder_recursive[i]<<"\n";
  }*/
  /*for(int i = 0; i < g->n_total; i++){
    if(preorder_recursive[i] != preorder[i]) {
      std::cout<<"preorder labels differ at vertex "<<i<<"\n";
      break;
    }
  }*/
  /*for(int i = 0; i < g->n_local; i++){
    printf("Rank %d: Vertex %d has a preorder label of %d \n",procid,g->local_unmap[i],preorder[i]);
  }
  for(int i = g->n_local; i < g->n_total; i++){
    printf("Rank %d: Vertex %d has a preorder label of %d \n",procid,g->ghost_unmap[i-g->n_local],preorder[i]);  
  }*/
  //while(true); 
  //3. calculate high and low values for each owned vertex
  uint64_t* lows = new uint64_t[g->n_total];
  uint64_t* highs = new uint64_t[g->n_total];
  calculate_high_low(g, comm, q, highs, lows, parents, preorder);
  //std::cout<<"Finished high low calculation\n";
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Creating Aux graph\n");
  }
  /*for(int i = 0; i < g->n_local; i++){
    printf("Rank %d: Vertex %d has a high of %d and a low of %d\n",procid,g->local_unmap[i],highs[i],lows[i]);
  }
  for(int i = g->n_local; i < g->n_total; i++){
    printf("Rank %d: Vertex %d has a high of %d and a low of %d\n",procid,g->ghost_unmap[i-g->n_local],highs[i],lows[i]);  
  }*/
  if(get_value(g->map, 0) < g->n_local) std::cout<<"vertex 0 has preorder label "<<preorder[get_value(g->map,0)]
                                                 <<", low "<<lows[get_value(g->map,0)]<<", high "<<highs[get_value(g->map,0)]
                                                 <<", n_desc "<<n_desc[get_value(g->map,0)]<<"\n";
  if(get_value(g->map, 8208) < g->n_local) std::cout<<"vertex 8208 has preorder label "<<preorder[get_value(g->map,8208)]
                                                    <<", low "<<lows[get_value(g->map,8208)]<<", high "<<highs[get_value(g->map,8208)]
                                                    <<", n_desc "<<n_desc[get_value(g->map,8208)]<<"\n";
  if(get_value(g->map, 24574) < g->n_local) std::cout<<"vertex 24574 has preorder label "<<preorder[get_value(g->map,24574)]
                                                 <<", low "<<lows[get_value(g->map,24574)]<<", high "<<highs[get_value(g->map,24574)]
                                                 <<", n_desc "<<n_desc[get_value(g->map,24574)]<<"\n";
  //4. create auxiliary graph.
  dist_graph_t* aux_g = new dist_graph_t;
  //std::map< std::pair<uint64_t, uint64_t> , uint64_t > edgeToAuxVert;
  //std::map< uint64_t, std::pair<uint64_t, uint64_t> > auxVertToEdge;
  create_aux_graph(g,comm,q,aux_g,preorder,lows,highs,n_desc,parents);
  //std::cout<<"Rank "<<procid<<"Finished creating auxiliary graph:\n";
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing Aux connectivity check\n");
  }
  
  /*for(int i = 0; i < aux_g->n_local; i++){
    std::pair<uint64_t, uint64_t> edge = auxVertToEdge[i];
    std::cout<<"\taux vertex {"<<edge.first<<","<<edge.second<<"}";//<<" is adjacent to:\n";
    int out_degree = out_degree(aux_g, i);
    std::cout<<" has degree "<<out_degree<<"\n";
    uint64_t* outs = out_vertices(aux_g, i);
    for(int j = 0; j < out_degree; j++){
      std::pair<uint64_t, uint64_t> otheredge = auxVertToEdge[outs[j]];
      std::cout<<"\t\taux vertex {"<<otheredge.first<<","<<otheredge.second<<"}";
      if(outs[j] >= aux_g->n_local){
        std::cout<<" owned by rank "<<aux_g->ghost_tasks[outs[j]-aux_g->n_local];
      }
      std::cout<<"\n";
    }
  }*/
  //5. flood-fill labels to determine which edges are in the same biconnected component, communicating the labels between copies of the same edge.
  //std::cout<<"aux_g->n_total = "<<aux_g->n_total<<"\n";
  //std::cout<<"aux_g->n_local = "<<aux_g->n_local<<"\n";
  uint64_t* labels = new uint64_t[aux_g->n_total];
  for(int i = 0; i < aux_g->n_total; i++) labels[i] = 0;
  connected_components(aux_g, comm, q, NULL,labels);
  //aux_connectivity_check(aux_g, edgeToAuxVert,auxVertToEdge, labels);
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing final edge labeling\n");
  }
  //std::cout<<"Rank "<<procid<<" finished propagating through aux graph\n";
  /*std::cout<<"Rank "<<procid<<"OUTPUTTING AUX GRAPH ********\n";
  for(int i = 0; i < aux_g->n_local; i++){
    std::pair<uint64_t, uint64_t> edge = auxVertToEdge[i];
    std::cout<<"Edge {"<<edge.first<<","<<edge.second<<"} is labeled "<<labels[i]<<"\n";
  }*/
  //6. remap labels to original edges and extend the labels to include certain non-tree edges that were excluded from the auxiliary graph.
  uint64_t* bicc_labels = new uint64_t[g->n_total];
  for(int i = 0; i < g->n_total; i++) bicc_labels[i] = NULL_KEY;
  finish_edge_labeling(g,srcs,aux_g,preorder,parents,labels,bicc_labels);
  //std::cout<<"Rank "<<procid<< " finished final edge labeling\n";
  //std::cout<<"aux_g->n = "<<aux_g->n_total<<" aux_g->m = "<<aux_g->m<<"\n";
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose && procid == 0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
  }
  /*std::cout<<"Rank "<<procid<<" OUTPUTTING FINAL LABELS *********\n";>
  for(int i = 0; i < g->n_local; i++){
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      uint64_t nbor = g->out_edges[j];
      if(nbor < g->n_local){
        std::cout<<"Rank "<<procid<<"Edge from "<<g->local_unmap[i]<<" to "<<g->local_unmap[nbor]<<" has label "<<bicc_labels[j]<<"\n";
      } else {
        std::cout<<"Rank "<<procid<<"Edge from "<<g->local_unmap[i]<<" to "<<g->ghost_unmap[nbor-g->n_local]<<" has label "<<bicc_labels[j]<<"\n";
      }
    }
  }*/
  //output the global list of articulation points, for easier validation (not getting rid of all debug output just yet.)
  uint64_t* artpts = new uint64_t[g->n_local];
  int n_artpts = 0;
  for(int i = 0; i < g->n_local; i++){
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      if(levels[i] < levels[g->out_edges[j]] && bicc_labels[i] != bicc_labels[g->out_edges[j]]){
        artpts[n_artpts++] = g->local_unmap[i];
        break;
      }
    }
  }
  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) sendcnts[i] = n_artpts;
  int* recvcnts = new int[nprocs];
  int status = MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);
  
  int* sdispls = new int[nprocs];
  int* rdispls = new int[nprocs];
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
  }

  int sendsize = 0;
  int recvsize = 0;
  for(int i = 0; i < nprocs; i++){
    sendsize += sendcnts[i];
    recvsize += recvcnts[i];
  }
  
  int * sendbuf = new int[sendsize];
  int* recvbuf = new int[recvsize];
  int sendbufidx = 0;
  for(int j = 0; j < nprocs; j++){
    for(int i = 0; i < n_artpts; i++){
      sendbuf[sendbufidx++] = artpts[i];
    }
  }
  status = MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);

  if(procid == 0){
    std::cout<<"Found "<<recvsize<<" artpts\n";
    /*for(int i = 0; i < recvsize; i++){
      std::cout<<"Vertex "<<recvbuf[i]<<" is an articulation point\n";
    }*/
  }

  if (verbose &&procid==0) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    //elt = timer();
    //printf("Doing BCC-LCA stage\n");
  }

  delete [] parents;
  delete [] levels;
  delete [] n_desc;
  delete [] preorder;
  delete [] lows;
  delete [] highs;
  //need to delete aux_g carefully.
  //delete [] labels;
  delete [] bicc_labels;
  delete [] artpts; 
  delete [] sendcnts;
  delete [] recvcnts;
  delete [] sendbuf;
  delete [] recvbuf;
  return 0;
}

extern "C" int create_dist_graph(dist_graph_t* g, 
          unsigned long n_global, unsigned long m_global, 
          unsigned long n_local, unsigned long m_local,
          unsigned long* local_adjs, unsigned long* local_offsets, 
          unsigned long* global_ids, unsigned long* vert_dist)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs > 1)
  {
    create_graph(g, (uint64_t)n_global, (uint64_t)m_global, 
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs, 
                 (uint64_t*)global_ids);
    relabel_edges(g, vert_dist);
  }
  else
  {
    create_graph_serial(g, (uint64_t)n_global, (uint64_t)m_global, 
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs);
  }

  //get_ghost_degrees(g);
  //get_ghost_weights(g);

  return 0;
}


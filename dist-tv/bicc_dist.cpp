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
#include <set>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <cstring>
#include <sys/time.h>
#include <time.h>
#include<iostream>
#include "bicc_dist.h"

#include "comms.h"
#include "dist_graph.h"
#include "bfs.h"

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
    sendbuf[sendbufidx++] = 0;
  }
  status = MPI_Alltoallv(sendbuf,sendcnts,sdispls,MPI_INT,recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);

  //fill in the values that were sent along with the GIDs
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx+= 2){
    int gid = recvbuf[exchangeIdx];
    int lid = get_value(g->map, gid);
    recvbuf[exchangeIdx + 1] = values[lid];
  }

  
  //communicate owned values back to ghosts
  status = MPI_Alltoallv(recvbuf,recvcnts,rdispls,MPI_INT, sendbuf,sendcnts,sdispls,MPI_INT,MPI_COMM_WORLD);

  for(int updateIdx = 0; updateIdx < sendsize; updateIdx+=2){
    int gid = sendbuf[updateIdx];
    int lid = get_value(g->map, gid);
    values[lid] = sendbuf[updateIdx+1];
  }
}

extern "C" int calculate_descendants(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* parents, bool* is_leaf, uint64_t* n_descendants){
  std::cout<<"Calculating Descendants\n";
  //std::set<uint64_t> unfinished;
  //all leaves have 1 descendant    
  for(uint64_t i = 0; i < g->n_total; i++){
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
        std::cout<<"\tchecking if vertex "<<parent<<" can be calculated\n"; 
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
    if(!global_done){
      //break;
      owned_to_ghost_value_comm(g,comm,q,n_descendants);
    }

  }
  return 0;
}

void calculate_preorder(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* levels,  uint64_t* parents, uint64_t* n_desc, uint64_t* preorder){
  std::cout<<"\tCalculating Preorder Labels\n";
  //go level-by-level on each process, communicating after each level is completed.
  int max_level = 0;
  for(int i = 0; i < g->n_local; i++){
    if(max_level < levels[i]) max_level = levels[i];
  }
  int global_max_level = 0;
  MPI_Allreduce(&max_level,&global_max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::cout<<"\t\tRank "<<procid<<"'s Max level = "<<max_level<<"\n";
  
  for(int level = 0; level <= global_max_level; level++){
    //std::cout<<"\t\tCalculating labels for level "<<level<<"\n";
    //look through all owned vertices and calculate all their children serially
    for(int i = 0; i < g->n_local; i++){
      if(levels[i] != level) continue;  //either not involved in current calculation, or not able to be calculated yet
      if(level == 0) {                   //base case, root is labeled 1
        preorder[i] = 1;
        continue;
      }
      int out_degree = out_degree(g, i);
      uint64_t* outs = out_vertices(g, i);
      uint64_t child_label = preorder[i] + 1; //first child's label is the parent's plus 1
      for(int j = 0; j < out_degree; j++){
        if(j != 0) {
          child_label += n_desc[outs[j-1]]; //the subsequent children need to factor in the previous childs' descendants
        }
        preorder[outs[j]] = child_label;
      }
    }
    std::cout<<"Rank "<<procid<<": entering communication\n"; 
    owned_to_ghost_value_comm(g,comm,q,preorder);
  }
  
}

void calculate_high_low(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, uint64_t* highs, uint64_t* lows, uint64_t* parents, uint64_t* preorder){
  
  for(int i = 0; i < g->n_total; i++){
    highs[i] = 0;
    lows[i] = 0;
  }
  
  //from leaves, compute highs and lows.
  uint64_t unfinished = g->n_total;
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
  }  
}

extern "C" int bicc_dist(dist_graph_t* g,mpi_data_t* comm, queue_data_t* q)
{

  double elt = 0.0, elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

  for(int i = 0; i < g->n_total; i++){
    int out_degree = out_degree(g, i);
    uint64_t* outs = out_vertices(g, i);
    printf("%d's neighbors:\n",i);
    for(int j = 0; j < out_degree; j++){
      printf("\t%d\n",outs[j]);
    }
  }
  uint64_t* parents = new uint64_t[g->n_total];
  uint64_t* levels = new uint64_t[g->n_total];
  bool* is_leaf = new bool[g->n_total];
  bicc_bfs_pull(g, comm, q, parents, levels, is_leaf, g->max_degree_vert);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  for(int i = 0; i < g->n_local; i++){
    int curr_global = g->local_unmap[i];
    printf("vertex %d, parent: %d, level: %d, is_leaf: %d\n",curr_global, parents[i], levels[i],is_leaf[i]);
  }
  for(int i = 0; i < g->n_ghost; i++){
    int curr = g->n_local + i;
    printf("vertex %d, parent: %d, level: %d, is_leaf: %d\n",g->ghost_unmap[i], parents[curr], levels[curr],is_leaf[curr]);
  }
  
  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing BCC-LCA stage\n");
  }

  
  
  //1. calculate descendants for each owned vertex, counting itself as its own descendant
  uint64_t* n_desc = new uint64_t[g->n_total];
  calculate_descendants(g,comm,q,parents,is_leaf,n_desc);
  std::cout<<"Finished calculating descendants\n";
  //2. calculate preorder labels for each owned vertex (using the number of descendants)
  delete [] is_leaf;
  uint64_t* preorder = new uint64_t[g->n_total];
  calculate_preorder(g,comm,q,levels,parents,n_desc,preorder); 
  std::cout<<"Finished calculating preorder labels\n";
  //3. calculate high and low values for each owned vertex
  uint64_t* lows = new uint64_t[g->n_total];
  uint64_t* highs = new uint64_t[g->n_total];
  calculate_high_low(g, comm, q, highs, lows, parents, preorder);
  std::cout<<"Finished high low calculation\n";
  //4. create vertices for each edge, including ghost edges. Ownership of ghost edges is ambiguous, but it shouldn't matter.

  //5. add edges to the auxiliary graph according to the high and low values

  //6. flood-fill labels to determine which edges are in the same biconnected component, communicating the labels between copies of the same edge.

  //7. extend the labels to include certain non-tree edges that were excluded from the auxiliary graph.

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    //elt = timer();
    //printf("Doing BCC-LCA stage\n");
  }

  delete [] parents;
  delete [] levels;

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


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
#include <map>
#include <queue>
#include <utility>
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


void communicate_preorder_labels(dist_graph_t* g, std::queue<uint64_t> &frontier, uint64_t* preorder){
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
    if(preorder[lid] == 0 && recvbuf[exchangeIdx + 1] != 0){
      preorder[lid] = recvbuf[exchangeIdx+1];
      frontier.push(lid);
      //std::cout<<"Rank "<<procid<<" pushing vertex "<<gid<<" onto the local queue\n";
    } else if (preorder[lid] != 0 && recvbuf[exchangeIdx+1]!=0 && preorder[lid] != recvbuf[exchangeIdx+1]) std::cout<<"*********sent and owned preorder labels mismatch*************\n";
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
      std::set<uint64_t> visited_nbors;
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
	if(currVert == 89194 || currVert == 471983) std::cout<<"Vertex "<<currVert<<" has "<<children<<"children and they have "<<n_desc<<" descedants, collectively\n";
        n_descendants[currVert] = n_desc + children_computed;
        //if(get_value(g->map, parents[currVert]) < g->n_local){
        //  otherQueue->push(get_value(g->map, parents[currVert]));
        //}
      } else {
        //std::cout<<"Rank "<<procid<<" was unable to compute n_desc for vertex "<<g->local_unmap[currVert]<<"\n";
        otherQueue->push(currVert);
      }
    }
    communicate_preorder_labels(g,*otherQueue, n_descendants);
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
      std::set<uint64_t> visited_nbors;
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
    communicate_preorder_labels(g,frontier,preorder);
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
  std::set<uint64_t> verts_in_queue; 
  //put all owned leaves on a frontier
  for(int i = 0; i < g->n_local; i++){
    primary_frontier.push(i);
    verts_in_queue.insert(i);
  }
  int global_done = 0;
  //while the number of vertices in the global queues is not zero
  while(!global_done){
    //while the primary queue is not empty
    while(currQueue->size() > 0){
      uint64_t currVert = currQueue->front();
      verts_in_queue.erase(currVert);
      std::cout<<"HIGH-LOW visiting vtx "<<currVert<<"\n";
      currQueue->pop();
      //if the vertex was previously computed, skip
      if(highs[currVert] != 0 && lows[currVert] != 0) continue;
      bool calculated = true;
      uint64_t low = preorder[currVert];
      uint64_t high = preorder[currVert];
      
      int out_degree = out_degree(g,currVert);
      uint64_t* outs = out_vertices(g,currVert);
      std::set<uint64_t> nbors_visited;
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
      //calculate the current vertex if possible, if not push it onto the secondary queue
      //if current vertex was calculated, push parent on the secondary frontier
      if(calculated){
        highs[currVert] = high;
        lows[currVert] = low;
        uint64_t local_parent = get_value(g->map, parents[currVert]);
        if(local_parent < g->n_local && verts_in_queue.count(local_parent) == 0){
          otherQueue->push(local_parent);
	  verts_in_queue.insert(local_parent);
        }
      } else {
        if(verts_in_queue.count(currVert) == 0){
	  otherQueue->push(currVert);
	  verts_in_queue.insert(currVert);
	}
      }
    }
    //communicate the highs and lows using the frontier comm function, secondary queue as input
    communicate_preorder_labels(g,*otherQueue,highs);
    communicate_preorder_labels(g,*otherQueue,lows);
    //see if secondary queues are all empty with an MPI_Allreduce, MPI_SUM
    int secondary = otherQueue->size();
    MPI_Allreduce(&secondary, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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

void create_aux_graph(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, dist_graph_t* aux_g, std::map<std::pair<uint64_t, uint64_t>, uint64_t> &edgeToAuxVert,
                      std::map<uint64_t, std::pair<uint64_t, uint64_t> > &auxVertToEdge, uint64_t* preorder, uint64_t* lows, uint64_t* highs, uint64_t* n_desc, uint64_t* parents){
  
  uint64_t n_srcs = 0;
  uint64_t n_dsts = 0;
  std::cout<<"Calculating the number of edges in the aux graph\n";  
  //count self edges for each tree edge (maybe nontree edges are necessary as well, maybe not. We'll see).
  for(uint64_t v = 0; v < g->n_local; v++){
    std::set<uint64_t> nbors_visited;
    for(uint64_t w_idx = g->out_degree_list[v]; w_idx < g->out_degree_list[v+1]; w_idx++){
      uint64_t w = g->out_edges[w_idx];
      if(nbors_visited.count(w) >0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(parents[v] == w_global || parents[w] == g->local_unmap[v]){
        //add a self edge in srcs and dsts
        n_srcs++;
        n_dsts++;
      }
    }
  }
  //non self-edges
  for(uint64_t v = 0; v < g->n_local; v++){
    std::set<uint64_t> nbors_visited;
    for(uint64_t j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; j++){
      uint64_t w = g->out_edges[j];
      if(nbors_visited.count(w) > 0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(parents[w] != g->local_unmap[v] && parents[v] != w_global){ //nontree edge
        if(preorder[v] + n_desc[v] <= preorder[w] || preorder[w] + n_desc[w] <= preorder[v]){
          n_srcs += 2;
          n_dsts += 2;
          //add {{parents[v], v}, {parents[w], w}} (undirected edges)
        }
      } else{
        if(parents[w] == g->local_unmap[v]){
          if(preorder[v] != 1 && (lows[w] < preorder[v] || highs[w] >= preorder[v] + n_desc[v])){
            //add {{parents[v], v} , {v, w}} as an edge
            n_srcs += 2;
            n_dsts += 2;
          }
        } else if (parents[v] == w_global){
          if(preorder[w] != 1 && (lows[v] < preorder[w] || highs[v] >= preorder[w] + n_desc[w])){
            //add {{parents[w], w}, {w, v}} as an edge
            n_srcs += 2;
            n_dsts += 2;
          }
        }
      }
    }
  }
  
  uint64_t** srcs = new uint64_t*[n_srcs];
  uint64_t** dsts = new uint64_t*[n_dsts];
  uint64_t** orig = new uint64_t*[n_srcs];
  //entries in these arrays take the form {v, w}, where v and w are global vertex identifiers.
  //additionally we ensure v < w.
  for(uint64_t i = 0; i < n_srcs; i++){
    srcs[i] = new uint64_t[2];
    dsts[i] = new uint64_t[2];
    orig[i] = new uint64_t[2];
  }
  uint64_t srcIdx = 0;
  uint64_t dstIdx = 0;
  std::cout<<"Creating srcs and dsts arrays\n";
  //self edges
  for(uint64_t v = 0; v < g->n_local; v++){
    std::set<uint64_t> nbors_visited;
    for(uint64_t w_idx = g->out_degree_list[v]; w_idx < g->out_degree_list[v+1]; w_idx++){
      uint64_t w = g->out_edges[w_idx];
      if(nbors_visited.count(w) >0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(parents[v] == w_global || parents[w] == g->local_unmap[v]){
        //add a self edge in srcs and dsts
        orig[srcIdx][0]=v;
        orig[srcIdx][1]=w;
        if(g->local_unmap[v] < w_global){
          srcs[srcIdx][0] = g->local_unmap[v];
          srcs[srcIdx++][1] = w_global;
          dsts[dstIdx][0] = g->local_unmap[v];
          dsts[dstIdx++][1] = w_global;
        } else {
          srcs[srcIdx][0] = w_global;
          srcs[srcIdx++][1] = g->local_unmap[v];
          dsts[dstIdx][0] = w_global;
          dsts[dstIdx++][1] = g->local_unmap[v];
        }
      }
    }
  }
  

  //non-self edges
  for(uint64_t v = 0; v < g->n_local; v++){
    std::set<uint64_t> nbors_visited;
    for(uint64_t j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; j++){
      uint64_t w = g->out_edges[j];//w is local, not preorder
      if(nbors_visited.count(w) > 0) continue;
      nbors_visited.insert(w);
      uint64_t w_global = 0;
      if(w < g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(parents[w] != g->local_unmap[v] && parents[v] != w_global){ //nontree edge
        if(preorder[v] + n_desc[v] <= preorder[w] || preorder[w] + n_desc[w] <= preorder[v]){
          //add {{parents[v], v}, {parents[w], w}} (undirected edges)
	  //std::cout<<"Rank "<<procid<<" adding {"<<parents[v]<<","<<g->local_unmap[v]<<"} -- {"<<parents[w]<<","<<w_global<<"} as an edge\n";
          orig[srcIdx][0] = v;
          orig[srcIdx][1] = w;
          if(parents[v] < g->local_unmap[v]){
            srcs[srcIdx][0] = parents[v];
            srcs[srcIdx++][1] = g->local_unmap[v];
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          } else {
            srcs[srcIdx][0] = g->local_unmap[v];
            srcs[srcIdx++][1] =  parents[v];
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          }
          if(parents[w] < w_global){
            dsts[dstIdx][0] = parents[w];
            dsts[dstIdx++][1] = w_global;
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          } else {
            dsts[dstIdx][0] = w_global;
            dsts[dstIdx++][1] = parents[w];
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          }
          orig[srcIdx][0] = v;
          orig[srcIdx][1] = w;
          //add {parents[w], w} as a src and {parents[v], v} as a dst
          if(parents[v] < g->local_unmap[v]){
            dsts[dstIdx][0] = parents[v];
            dsts[dstIdx++][1] = g->local_unmap[v];
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          } else {
            dsts[dstIdx][0] = g->local_unmap[v];
            dsts[dstIdx++][1] =  parents[v];
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          }
          if(parents[w] < w_global){
            srcs[srcIdx][0] = parents[w];
            srcs[srcIdx++][1] = w_global;
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          } else {
            srcs[srcIdx][0] = w_global;
            srcs[srcIdx++][1] = parents[w];
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          }
        }
      } else{
        if(parents[w] == g->local_unmap[v]){
          if(preorder[v] != 1 && (lows[w] < preorder[v] || highs[w] >= preorder[v] + n_desc[v])){
            //add {{parents[v], v} , {v, w}} as an edge
            orig[srcIdx][0] = v;
            orig[srcIdx][1] = w;
            //std::cout<<"Rank "<<procid<<" adding {"<<parents[v]<<","<<g->local_unmap[v]<<"} -- {"<< g->local_unmap[v]<<","<<w_global<<"} as an edge\n";
            if(parents[v] < g->local_unmap[v]){
              srcs[srcIdx][0] = parents[v];
              srcs[srcIdx++][1] = g->local_unmap[v];
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            } else {
              srcs[srcIdx][0] = g->local_unmap[v];
              srcs[srcIdx++][1] = parents[v];
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            }
            if(g->local_unmap[v] < w_global){
              dsts[dstIdx][0] = g->local_unmap[v];
              dsts[dstIdx++][1]=w_global;
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            } else {
              dsts[dstIdx][0] = w_global;
              dsts[dstIdx++][1] = g->local_unmap[v];
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            }
            orig[srcIdx][0] = v;
            orig[srcIdx][1] = w;
            //add {v, w} as a src and {parents[v], v} as a dst
            if(parents[v] < g->local_unmap[v]){
              dsts[dstIdx][0] = parents[v];
              dsts[dstIdx++][1] = g->local_unmap[v];
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            } else {
              dsts[dstIdx][0] = g->local_unmap[v];
              dsts[dstIdx++][1] = parents[v];
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            }
            if(g->local_unmap[v] < w_global){
              srcs[srcIdx][0] = g->local_unmap[v];
              srcs[srcIdx++][1]=w_global;
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            } else {
              srcs[srcIdx][0] = w_global;
              srcs[srcIdx++][1] = g->local_unmap[v];
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            }
          }
        } else if (parents[v] == w_global){
          if(preorder[w] != 1 && (lows[v] < preorder[w] || highs[v] >= preorder[w] + n_desc[w])){
            //add {{parents[w], w}, {w, v}} as an edge
            //std::cout<<"Rank "<<procid<<" adding {"<<parents[w]<<","<<w_global<<"} -- {"<< w_global <<","<<g->local_unmap[v]<<"} as an edge\n";
            orig[srcIdx][0] = v;
            orig[srcIdx][1] = w;
            if(parents[w] < w_global){
              srcs[srcIdx][0] = parents[w];
              srcs[srcIdx++][1] = w_global;
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            } else {
              srcs[srcIdx][0] = w_global;
              srcs[srcIdx++][1] = parents[w];
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            }
            if(g->local_unmap[v] < w_global){
              dsts[dstIdx][0] = g->local_unmap[v];
              dsts[dstIdx++][1]=w_global;
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            } else {
              dsts[dstIdx][0] = w_global;
              dsts[dstIdx++][1] = g->local_unmap[v];
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            }
            orig[srcIdx][0] = v;
            orig[srcIdx][1] = w;
            //add {v, w} as a src and {parents[v], v} as a dst
            if(parents[w] < w_global){
              dsts[dstIdx][0] = parents[w];
              dsts[dstIdx++][1] = w_global;
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            } else {
              dsts[dstIdx][0] = w_global;
              dsts[dstIdx++][1] = parents[w];
              //std::cout<<"Rank "<<procid<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
            }
            if(g->local_unmap[v] < w_global){
              srcs[srcIdx][0] = g->local_unmap[v];
              srcs[srcIdx++][1]=w_global;
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            } else {
              srcs[srcIdx][0] = w_global;
              srcs[srcIdx++][1] = g->local_unmap[v];
              //std::cout<<"Rank "<<procid<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
            }
          }
        }
      }
    }
  }
  std::cout<<"Finished populating srcs and dsts arrays:\n";
  
  /*for(int i = 0; i < n_srcs; i++){
    std::cout<<"Rank "<<procid<<"("<<srcs[i][0]<<","<<srcs[i][1]<<") -- ("<<dsts[i][0]<<","<<dsts[i][1]<<")\n";
  }*/
  //srcs and dsts contain all edges in the auxiliary graph, copy other csr creation function to turn it into a graph.
  // remember to collect ghosting information, can loop through srcs and dsts to label auxiliary vertices and determine which ones are ghosts.
   
  // assign vertex identifiers to edge pairs, and map identifiers back to edge pairs.
  // simultaneously, copy ghosting information from the edges.
  //std::map< std::pair<uint64_t, uint64_t>, uint64_t> edgeToAuxVert;
  //std::map< uint64_t, std::pair<uint64_t, uint64_t> > auxVertToEdge;
  
  //label all locally owned auxiliary vertices, they are contained in one of srcs or dsts.
  //keep a running counter, label the ghosts afterwards.
  uint64_t currId = 0;
  for(uint64_t i = 0; i < srcIdx; i++){
    std::pair<uint64_t, uint64_t> auxVert = std::make_pair(srcs[i][0], srcs[i][1]);
    if( edgeToAuxVert.find(auxVert) == edgeToAuxVert.end()) {
      //insert vert into both maps if it should not be ghosted.
      if((get_value(g->map,auxVert.first) < g->n_local && get_value(g->map,auxVert.second) < g->n_local) || (orig[i][0] < g->n_local && orig[i][1] < g->n_local) ) {// both are not ghosts (good!)
        edgeToAuxVert.insert( std::make_pair(auxVert, currId));
        auxVertToEdge.insert( std::make_pair(currId, auxVert));
        currId++;
      } else  { //one of the endpoints is a ghosted vertex
        bool firstIsGhost = get_value(g->map,auxVert.first) >= g->n_local;
        bool secondIsGhost = get_value(g->map,auxVert.second) >= g->n_local;
         
        //only one endpoint can be ghosted, we don't have edges from ghosts to ghosts.
        if(auxVert.second < auxVert.first && secondIsGhost && !firstIsGhost){ //this is locally owned
          edgeToAuxVert.insert(std::make_pair(auxVert, currId));
          auxVertToEdge.insert(std::make_pair(currId, auxVert));
          currId++;
        } else if (auxVert.second > auxVert.first && firstIsGhost && !secondIsGhost){ //this is locally owned
          edgeToAuxVert.insert(std::make_pair(auxVert, currId));
          auxVertToEdge.insert(std::make_pair(currId, auxVert));
          currId++;
        }
        //don't add anything for ghosted vertices
      }
    }
  }
  delete [] orig;
  uint64_t aux_n_local = currId;
  //loop back through the arrays and add the ghosts to the indices  
  for(uint64_t i = 0; i < srcIdx; i++){
    std::pair<uint64_t, uint64_t> auxVert = std::make_pair(srcs[i][0], srcs[i][1]);
    if(edgeToAuxVert.find(auxVert) == edgeToAuxVert.end()){ //the edge doesn't already exist, so it's ghosted
      edgeToAuxVert.insert(std::make_pair(auxVert,currId));
      auxVertToEdge.insert(std::make_pair(currId,auxVert));
      currId++;
    }
    //std::cout<<"Edge {"<<auxVert.first<<","<<auxVert.second<<"} mapped to local ID: "<<edgeToAuxVert[auxVert]<<"\n";
  }

  uint64_t aux_n_ghosts = currId - aux_n_local;
  uint64_t aux_n_total = currId;
  
  //everything has a numeric label, start creating the csr
  aux_g->n_total = currId;
  aux_g->n_local = aux_n_local;
  aux_g->n_ghost = aux_n_ghosts;      
    
  uint64_t* aux_degrees = new uint64_t[aux_n_local];
  uint64_t* aux_offsets = new uint64_t[aux_n_local+1];
  for(uint64_t i = 0; i < aux_n_local+1; i++) aux_offsets[i] = 0;
  for(uint64_t i = 0; i < aux_n_local; i++) aux_degrees[i] = 0;
  uint64_t* ghost_tasks = new uint64_t[aux_n_ghosts];
  for(uint64_t i = 0; i < aux_n_ghosts; i ++) ghost_tasks[i] = 0;

  //count degrees and set up ghost tasks.
  for(uint64_t i = 0; i < srcIdx; i++){
    std::pair<uint64_t, uint64_t> auxVert = std::make_pair(srcs[i][0], srcs[i][1]);
    uint64_t auxVertId = edgeToAuxVert[auxVert];
    if(auxVertId < aux_n_local){
      aux_degrees[auxVertId]++;
    } else {
      //std::cout<<"Ghost Edge {"<<auxVert.first<<","<<auxVert.second<<"} mapped to ID: "<<edgeToAuxVert[auxVert]<<"\n";
      if(get_value(g->map, auxVert.second) != NULL_KEY && get_value(g->map,auxVert.second) >= g->n_local){
        //std::cout<<"Aux Vertex "<<auxVertId<<" is owned by rank "<<g->ghost_tasks[get_value(g->map,auxVert.second)-g->n_local]<<"\n";
        ghost_tasks[auxVertId-aux_n_local] = g->ghost_tasks[get_value(g->map, auxVert.second) - g->n_local];
      } else if (get_value(g->map,auxVert.first) >= g->n_local){
        //std::cout<<"Aux Vertex "<<auxVertId<<" is owned by rank "<<g->ghost_tasks[get_value(g->map,auxVert.first)-g->n_local]<<"\n";
        ghost_tasks[auxVertId-aux_n_local] = g->ghost_tasks[get_value(g->map, auxVert.first) - g->n_local];
      }
      /*if(auxVert.first < g->n_local){
        ghost_tasks[auxVertId] = g->ghost_tasks[get_value(g->map,auxVert.second) - g->n_local];
      } else {
        ghost_tasks[auxVertId] = g->ghost_tasks[get_value(g->map,auxVert.first) - g->n_local];
      }*/
    }
  }

  /*for(int i =0; i<aux_n_local; i++){
    std::cout<<"aux vertex "<<i<<" has degree "<<aux_degrees[i]<<"\n";
  }*/
  //set up the offsets
  aux_offsets[0] = 0;
  for(uint64_t i = 1; i < aux_n_local+1; i++){
    aux_offsets[i] = aux_degrees[i-1] + aux_offsets[i-1];
  }
  
  uint64_t* aux_adjacencies = new uint64_t[n_srcs];
  uint64_t* aux_edges_added = new uint64_t[aux_n_local];
  for(uint64_t i = 0; i < aux_n_local; i++) aux_edges_added[i] = 0;
  for(uint64_t i = 0; i < n_srcs; i++) aux_adjacencies[i] = 0;
  //set up the adjacency array
  for(uint64_t i = 0; i < srcIdx; i++){
    std::pair<uint64_t, uint64_t> srcEdge = std::make_pair(srcs[i][0], srcs[i][1]);
    std::pair<uint64_t, uint64_t> dstEdge = std::make_pair(dsts[i][0], dsts[i][1]);
    uint64_t srcVtx = edgeToAuxVert[srcEdge];
    uint64_t dstVtx = edgeToAuxVert[dstEdge];
    if(srcVtx < aux_n_local){ //make sure only non ghosted vertices get outgoing edges
      aux_adjacencies[aux_offsets[srcVtx] + aux_edges_added[srcVtx]] = dstVtx;
      aux_edges_added[srcVtx]++;
    }
  }
  aux_g->m = n_srcs;
  aux_g->m_local = n_srcs;
  aux_g->out_degree_list = aux_offsets;
  aux_g->out_edges = aux_adjacencies;
  aux_g->ghost_tasks = ghost_tasks;
  delete [] srcs;
  delete [] dsts;
}

void aux_graph_comm(dist_graph_t* aux_g, std::map< std::pair<uint64_t, uint64_t>, uint64_t > edgeToAuxVert, std::map< uint64_t, std::pair<uint64_t, uint64_t> > auxVertToEdge,
                    uint64_t* labels, std::queue<uint64_t>& frontier){
  
  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendcnts[i] = 0;
  }
  
  for(int i = aux_g->n_local; i < aux_g->n_total; i++){
    sendcnts[aux_g->ghost_tasks[i-aux_g->n_local]] += 3; //one for each endpoint, one for a label
  }

  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) recvcnts[i] = 0;
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
  int* sentcount = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendsize += sendcnts[i];
    recvsize += recvcnts[i];
    sentcount[i] = 0;
  }
  
  int* sendbuf = new int[sendsize];
  int* recvbuf = new int[recvsize];
  
  //go through all the ghosted vertices, send their GIDs to the owning processor
  for(int i = aux_g->n_local; i < aux_g->n_total; i++){
    int proc_to_send = aux_g->ghost_tasks[i-aux_g->n_local];
    int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send];
    sentcount[proc_to_send] += 3;
    std::pair<uint64_t, uint64_t> globalEdge = auxVertToEdge[i];
    sendbuf[sendbufidx++] = globalEdge.first;
    sendbuf[sendbufidx++] = globalEdge.second;
    sendbuf[sendbufidx++] = labels[i];
    //std::cout<<"Rank "<<procid<<" sending {"<<globalEdge.first<<","<<globalEdge.second<<"} to rank "<<proc_to_send<<" with label "<<labels[i]<<"\n";
  }

  status = MPI_Alltoallv(sendbuf,sendcnts,sdispls,MPI_INT,recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);
  
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx += 3){
    std::pair<uint64_t, uint64_t> edge = std::make_pair(recvbuf[exchangeIdx], recvbuf[exchangeIdx+1]);
    //std::cout<<"Rank "<<procid<<" received {"<<recvbuf[exchangeIdx]<<","<<recvbuf[exchangeIdx+1]<<"} with label "<<recvbuf[exchangeIdx+2]<<"\n";
    if(edgeToAuxVert.find(edge) != edgeToAuxVert.end()){
      uint64_t vert = edgeToAuxVert[edge];
      uint64_t owned_label = labels[vert];
      uint64_t recvd_label = recvbuf[exchangeIdx+2];
      if(recvd_label != 0 && owned_label == 0 && vert < aux_g->n_local){
        //std::cout<<"Rank "<<procid<<" pushing {"<<edge.first<<","<<edge.second<<"} with label "<<recvd_label<<"\n";
        frontier.push(vert);
      } else if (recvd_label == 0){
        recvbuf[exchangeIdx + 2] = owned_label;
      } else if (recvd_label != owned_label){
        std::cout<<"a received label doesn't match the owned label, this should not happen\n";
      }
    }
  }

  status = MPI_Alltoallv(recvbuf, recvcnts, rdispls, MPI_INT, sendbuf, sendcnts, sdispls, MPI_INT, MPI_COMM_WORLD);
  //this is back where the verts are ghosted, should not attempt to add to frontier.
  for(int updateIdx = 0; updateIdx < sendsize; updateIdx += 3){
    std::pair<uint64_t, uint64_t> edge = std::make_pair(sendbuf[updateIdx], sendbuf[updateIdx+1]);
    uint64_t vert = edgeToAuxVert[edge];
    uint64_t owned_label = labels[vert];
    uint64_t recvd_label = sendbuf[updateIdx +2];
    if(owned_label == 0){
      labels[vert] = recvd_label;
      //std::cout<<"Rank "<<procid<<" setting {"<<edge.first<<","<<edge.second<<"} to have label "<<recvd_label<<"\n";
    } else if (recvd_label != 0 && owned_label != recvd_label){
      std::cout<<"owned and received labels mismatch, this should not happen\n";
    }
  }
  delete [] sendcnts;
  delete [] recvcnts;
  delete [] sdispls;
  delete [] rdispls;
  delete [] sentcount;
  delete [] sendbuf;
  delete [] recvbuf;
}

void aux_connectivity_check(dist_graph_t* aux_g, std::map< std::pair<uint64_t, uint64_t>, uint64_t> edgeToAuxVert, std::map<uint64_t, std::pair<uint64_t, uint64_t> > auxVertToEdge, uint64_t* labels){
  
  //initialize all the labels to zero
  for(int i = 0; i < aux_g->n_total; i++){
    labels[i] = 0;
  }
  
  //loop until there are no unlabeled vertices anywhere
  int local_done = 0;
  int global_done = 0;
  int round = 1;
  while(!global_done){ 
    //collectively determine which process will start the propagation
    int n_unlabeled = 0;
    for(int i = 0; i < aux_g->n_local; i++){
      if(labels[i] == 0) n_unlabeled++;
    }
    //std::cout<<"Rank "<<procid<<" #unlabeled = "<<n_unlabeled<<"\n"; 
    int propProc = -1;
    if(n_unlabeled > 0) propProc = procid;
    int global_propProc = -1;
    MPI_Allreduce(&propProc, &global_propProc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::cout<<"Rank "<<procid<<" global_propProc = "<<global_propProc<<"\n";
    //the selected process propagates from any unlabeled vertex, creating a frontier as it goes.
    std::queue<uint64_t> frontier;
    std::set<uint64_t> verts_in_frontier;
    if(global_propProc == procid){
      for(int i = 0; i < aux_g->n_local; i++){
        if(labels[i] == 0){
          frontier.push(i);
	  verts_in_frontier.insert(i);
          break;
        }
      }
    }
    uint64_t currLabel = global_propProc + nprocs*round;
  //isn't   //while this propagation is not finished, do local work
    
    while(!global_done){
      while(frontier.size() > 0){
        uint64_t currVert = frontier.front();
        frontier.pop();
	verts_in_frontier.erase(currVert);
        if(labels[currVert] == 0){
          labels[currVert] = currLabel;
          std::pair<uint64_t, uint64_t> edge = auxVertToEdge[currVert];
          std::cout<<"Rank "<<procid<<" labeled {"<<edge.first<<", "<<edge.second<<"} with label "<<currLabel<<"\n";
        }
        if(currVert < aux_g->n_local){ //we don't have adjacency info for ghosted verts
          int out_degree = out_degree(aux_g, currVert);
          uint64_t* outs = out_vertices(aux_g, currVert);
          for(int j = 0; j < out_degree; j++){
            if(labels[outs[j]] == 0 && verts_in_frontier.count(outs[j]) == 0) {
              //std::cout<<"Rank "<<procid<<" adding vertex "<<outs[j]<<" to the frontier\n";
              frontier.push(outs[j]);
	      verts_in_frontier.insert(outs[j]);
            }
          }
        }
	std::cout<<"Rank "<<procid<<" frontier.size() = "<<frontier.size()<<"\n";
      }
      //once the frontier is empty, communicate to all processes using the aux_communicate function
      //std::cout<<"Rank "<<procid<<" is starting communication\n";
      aux_graph_comm(aux_g,edgeToAuxVert,auxVertToEdge,labels, frontier);
      local_done = frontier.size();
      global_done = 0;
      //std::cout<<"Rank "<<procid<<" starting communication\n";
      MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      std::cout<<"Rank "<<procid<<" number of vertices in the frontier "<<global_done<<"\n";
      //if all frontiers are empty after communicating, this propagation is finished
      global_done = !global_done;
    }
    round++;
    n_unlabeled = 0;
    for(int i = 0; i < aux_g->n_local; i++){
      if(labels[i] == 0) n_unlabeled++;
    }
    MPI_Allreduce(&n_unlabeled, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout<<"number of unlabeled vertices left in the global graph "<<global_done<<"\n";
    //if there are no unlabeled vertices anywhere, the entire process is done.
    global_done = !global_done;
  }
}

void finish_edge_labeling(dist_graph_t* g, dist_graph_t* aux_g, std::map< std::pair<uint64_t, uint64_t>, uint64_t> & edgeToAuxVert, 
			  std::map< uint64_t, std::pair<uint64_t, uint64_t> > & auxVertToEdge, uint64_t* preorder, uint64_t* parents,
                          uint64_t * aux_labels, uint64_t* final_labels){

  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) sendcnts[i] = 0;
  //map the aux vertices to edges, and apply the corresponding aux_labels to the final edge labels.
  for(uint64_t vert = 0; vert < g->n_local; vert++){
    for(uint64_t nbor_idx = g->out_degree_list[vert]; nbor_idx < g->out_degree_list[vert+1]; nbor_idx++){
      uint64_t nbor_local = g->out_edges[nbor_idx];
      uint64_t nbor_global = 0;
      uint64_t vert_global = g->local_unmap[vert];
      
      if(nbor_local < g->n_local) nbor_global = g->local_unmap[nbor_local];
      else nbor_global = g->ghost_unmap[nbor_local - g->n_local];
      std::pair<uint64_t, uint64_t> edge;
      if(nbor_global < vert_global){
        edge.first = nbor_global;
        edge.second = vert_global;
      } else if (vert_global < nbor_global){
        edge.first = vert_global;
        edge.second = nbor_global;
      }
      //first, check to see if the global edge is in the aux graph map.
      if(edgeToAuxVert.find(edge) != edgeToAuxVert.end()){
        uint64_t auxVert = edgeToAuxVert[edge];
        if(aux_labels[auxVert] != 0) final_labels[nbor_idx] = aux_labels[auxVert];
      //for each nontree edge {v,w}
      } else if(parents[vert] != nbor_global && parents[nbor_local] != g->local_unmap[vert]){
        //such that preorder[v] < preorder[w]
        if(preorder[vert] < preorder[nbor_local]){
          //nbor_local is w
          //if w is not ghosted, label the edge {v,w} the same as {parents[w], w} 
          if(nbor_local < g->n_local){
            std::pair<uint64_t, uint64_t> parent_edge;
            if(parents[nbor_local] < nbor_global){
              parent_edge.first = parents[nbor_local];
              parent_edge.second = nbor_global;
            } else {
              parent_edge.first = nbor_global;
              parent_edge.second = parents[nbor_local];
            }
            final_labels[nbor_idx] = aux_labels[edgeToAuxVert[parent_edge]];
            //std::cout<<"Labeling edge from "<<vert_global<<" to "<<nbor_global<<" with label "<<final_labels[nbor_idx]<<"\n";
            //need to also set the reverse edge? We consider both cases, so maybe not necessary.
          } else {
            //if w is ghosted, build up sendcnts to request the relevant edge label from the owning process
            sendcnts[g->ghost_tasks[nbor_local - g->n_local]]+=3;

          }
        } else if (preorder[nbor_local] < preorder[vert]){
          //vert is w
          //if w is not ghosted, label the edge {v,w the same as {parents[w], w]
          if(vert < g->n_local){
            std::pair<uint64_t, uint64_t> parent_edge;
            if(parents[vert] < vert_global){
              parent_edge.first = parents[vert];
              parent_edge.second = vert_global;
            } else {
              parent_edge.first = vert_global;
              parent_edge.second = parents[vert];
            }
            final_labels[nbor_idx] = aux_labels[edgeToAuxVert[parent_edge]];
            //std::cout<<"Labeling edge from "<<vert_global<<" to "<<nbor_global<<" with label "<<final_labels[nbor_idx]<<"\n";
          } else {
            //add to the sendcnts, not sure this can actually happen
            sendcnts[g->ghost_tasks[vert - g->n_local]]+=3;
          }
        }
      }
    }
  }
  //using sendcnts, build up arrays for a collective communication
  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) recvcnts[i] = 0;
  int status = MPI_Alltoall(sendcnts,1,MPI_INT, recvcnts,1,MPI_INT,MPI_COMM_WORLD);
  
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
  
  for(uint64_t vert = 0; vert < g->n_local; vert++){
    for(uint64_t nbor_idx = g->out_degree_list[vert]; nbor_idx < g->out_degree_list[vert+1]; nbor_idx++){
      uint64_t nbor_local = g->out_edges[nbor_idx];
      uint64_t nbor_global = 0;
      uint64_t vert_global = g->local_unmap[vert];
      
      if(nbor_local < g->n_local) nbor_global = g->local_unmap[nbor_local];
      else nbor_global = g->ghost_unmap[nbor_local - g->n_local];
      std::pair<uint64_t, uint64_t> edge;
      if(nbor_global < vert_global){
        edge.first = nbor_global;
        edge.second = vert_global;
      } else if (vert_global < nbor_global){
        edge.first = vert_global;
        edge.second = nbor_global;
      }
      if(edgeToAuxVert.find(edge) == edgeToAuxVert.end() && parents[vert] != nbor_global && parents[nbor_local] != g->local_unmap[vert]){
        if(preorder[vert] < preorder[nbor_local] && nbor_local >= g->n_local){
          int owning_proc = g->ghost_tasks[nbor_local - g->n_local];
          int sendbufidx = sdispls[owning_proc] + sentcount[owning_proc];
          std::pair<uint64_t, uint64_t> parent_edge;
          if(parents[nbor_local] < nbor_global){
            parent_edge.first = parents[nbor_local];
            parent_edge.second = nbor_global;
          } else {
            parent_edge.first = nbor_global;
            parent_edge.second = parents[nbor_local];
          }
          sentcount[owning_proc] += 3;
          sendbuf[sendbufidx++] = parent_edge.first;
          sendbuf[sendbufidx++] = parent_edge.second;
          sendbuf[sendbufidx++] = 0;
        } else if (preorder[nbor_local] < preorder[vert] && vert >= g->n_local ){
          //not sure this will actually happen
          int owning_proc = g->ghost_tasks[vert - g->n_local];
          int sendbufidx = sdispls[owning_proc] + sentcount[owning_proc];
          std::pair<uint64_t, uint64_t> parent_edge;
          if(parents[vert] < vert_global){
            parent_edge.first = parents[vert];
            parent_edge.second = vert_global;
          } else {
            parent_edge.first = vert_global;
            parent_edge.second = parents[vert];
          }
          sentcount[owning_proc] += 3;
          sendbuf[sendbufidx++] = parent_edge.first;
          sendbuf[sendbufidx++] = parent_edge.second;
          sendbuf[sendbufidx++] = 0;
        } 
      }
    }
  }
  //communicate!
  status = MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);
  //set labels to send back
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx += 3){
    //look up the edges in the first and second indices of the recvbuf
    std::pair<uint64_t, uint64_t> edge_to_lookup;
    edge_to_lookup.first = recvbuf[exchangeIdx];
    edge_to_lookup.second = recvbuf[exchangeIdx+1];
    uint64_t edge_label = aux_labels[edgeToAuxVert[edge_to_lookup]];
    recvbuf[exchangeIdx+2] = edge_label;
    //set the label in the third index of the recvbuf to be the label of the edges
  }
  //send the labels back and finish labeling everything.
  status = MPI_Alltoallv(recvbuf, recvcnts, rdispls, MPI_INT, sendbuf, sendcnts, sdispls, MPI_INT, MPI_COMM_WORLD);

  //for each local edge {vert, nbor} in the graph (we disregard the direction and so label both directed edges if they exist)
  for(int vert = 0; vert < g->n_local; vert++){
    for(int edgeIdx = g->out_degree_list[vert]; edgeIdx < g->out_degree_list[vert+1]; edgeIdx++){
      uint64_t vert_global = g->local_unmap[vert];
      uint64_t nbor = g->out_edges[edgeIdx];
      uint64_t nbor_global = 0;
      if(nbor < g->n_local){
        nbor_global = g->local_unmap[nbor];
      } else {
        nbor_global = g->ghost_unmap[nbor - g->n_local];
      }
      //if {vert, nbor} is nontree
      if(parents[vert] != nbor_global && parents[nbor] != vert_global){
        //if preorder[vert] < preorder[nbor] and the local edge is not labeled
        if(preorder[vert] < preorder[nbor] && final_labels[edgeIdx] == 0 ){
          //look through edges received to find the relevant edge {parents[nbor], g->local_unmap[nbor]} and use its label
          for(int updateIdx = 0; updateIdx < sendsize; updateIdx+=3){
            std::pair<uint64_t, uint64_t> update_edge;
            update_edge.first = sendbuf[updateIdx];
            update_edge.second = sendbuf[updateIdx+1];
            uint64_t update_label = sendbuf[updateIdx+2];
            if((update_edge.first == nbor_global || update_edge.second == nbor_global) && (update_edge.first == parents[nbor] || update_edge.second == parents[nbor])){
              final_labels[edgeIdx] = update_label;
              //std::cout<<"Update label = "<<update_label<<"\n";
              break; //only one edge received corresponds to the current edge
            }
          }
        //else if preorder[nbor] < preorder[vert] and the local edge is not labeled
        } else if (preorder[nbor] < preorder[vert] && final_labels[edgeIdx] == 0){ //again, not sure this will ever happen
          //look through edges received to find the relevant edge {parents[vert], g->local_unmap[vert]} and use its label
          for(int updateIdx = 0; updateIdx < sendsize; updateIdx +=3){
            std::pair<uint64_t, uint64_t> update_edge;
            update_edge.first = sendbuf[updateIdx];
            update_edge.second = sendbuf[updateIdx+1];
            uint64_t update_label = sendbuf[updateIdx+2];
            if((update_edge.first == vert_global || update_edge.second == vert_global) && (update_edge.first == parents[vert] || update_edge.second == parents[vert])){
              final_labels[edgeIdx] = update_label;
              break;
            }
          }
        }
      }
    }
  }
  

  for(int i = 0; i < nprocs; i++) sendcnts[i] = 0;
  //there are still a few unlabeled edges possible, so if there are any unlabeled edges, send their GID pairs to the other process that owns an endpoint
  for(int i = 0; i < g->n_local; i++){
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      uint64_t i_global = g->local_unmap[i];
      uint64_t nbor = g->out_edges[j];
      uint64_t nbor_global = 0;
      if(nbor < g->n_local) nbor_global = g->local_unmap[nbor];
      else nbor_global = g->ghost_unmap[nbor-g->n_local];
      //if the label of this edge is zero, send the gids to the process that owns the ghosted vertex.
      if(final_labels[j] == 0){
        if(nbor >= g->n_local){
          sendcnts[g->ghost_tasks[nbor - g->n_local]]+= 3;
        } else{
          printf("Rank %d has an unlabeled edge that is not ghosted somewhere else\n",procid);
        }
      }
    }
  }
  for(int i = 0; i < nprocs; i++) recvcnts[i] = 0;
  
  status = MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);

  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
  }
  
  sendsize = 0;
  recvsize = 0;
  for(int i = 0; i < nprocs; i++){
    sendsize += sendcnts[i];
    recvsize += recvcnts[i];
    sentcount[i] = 0;
  }

  delete [] sendbuf;
  delete [] recvbuf;
  sendbuf = new int[sendsize];
  recvbuf = new int[recvsize];

  //go through all unlabeled edges again, add their gids and 0 for their label.
  for(int i = 0; i < g->n_local; i++){
    for(int j= g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      uint64_t i_global = g->local_unmap[i];
      uint64_t nbor = g->out_edges[j];
      uint64_t nbor_global = 0;
      if(nbor < g->n_local) nbor_global = g->local_unmap[nbor];
      else nbor_global = g->ghost_unmap[nbor-g->n_local];
      if(final_labels[j] == 0){
        if(nbor >= g->n_local){
          int proc_to_send = g->ghost_tasks[nbor - g->n_local];
          int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send];
          sentcount[proc_to_send] += 3;
          sendbuf[sendbufidx++] = nbor_global; //it is owned on the other process, easier to find
          sendbuf[sendbufidx++] = i_global;
          sendbuf[sendbufidx++] = 0;
        }
      }
    }
  }
  status = MPI_Alltoallv(sendbuf,sendcnts,sdispls,MPI_INT,recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);

  //fill in the edge labels
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx += 3){
    uint64_t global_owned = recvbuf[exchangeIdx];
    uint64_t local_owned = get_value(g->map, global_owned);
    uint64_t global_ghost = recvbuf[exchangeIdx + 1];
    uint64_t local_ghost = get_value(g->map, global_ghost);
    
    //int out_degree = out_degree(g, local_owned);
    //uint64_t* outs = out_vertices(g, local_owned);
    for(int i = g->out_degree_list[local_owned]; i < g->out_degree_list[local_owned+1]; i++){
      if(g->out_edges[i] == local_ghost){
        recvbuf[exchangeIdx + 2] = final_labels[i];
      }
    }
  }
  
  status = MPI_Alltoallv(recvbuf, recvcnts, rdispls, MPI_INT, sendbuf, sendcnts, sdispls, MPI_INT, MPI_COMM_WORLD);
  for(int updateIdx = 0; updateIdx < sendsize; updateIdx += 3){
    uint64_t global_ghost = sendbuf[updateIdx];
    uint64_t local_ghost = get_value(g->map, global_ghost);
    uint64_t global_owned = sendbuf[updateIdx+1];
    uint64_t local_owned = get_value(g->map, global_owned);
    
    for(int i = g->out_degree_list[local_owned]; i < g->out_degree_list[local_owned+1]; i++){
      if(g->out_edges[i] == local_ghost){
        final_labels[i] = sendbuf[updateIdx+2];
      }
    }
  }
  delete [] sendcnts;
  delete [] recvcnts;
  delete [] sdispls;
  delete [] rdispls;
  delete [] sentcount;
  delete [] sendbuf;
  delete [] recvbuf;
}

extern "C" int bicc_dist(dist_graph_t* g,mpi_data_t* comm, queue_data_t* q)
{

  double elt = 0.0, elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

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
  bicc_bfs_pull(g, comm, q, parents, levels, g->max_degree_vert);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  /*for(int i = 0; i < g->n_local; i++){
    int curr_global = g->local_unmap[i];
    printf("vertex %d, parent: %d, level: %d, is_leaf: %d\n",curr_global, parents[i], levels[i],is_leaf[i]);
  }
  for(int i = 0; i < g->n_ghost; i++){
    int curr = g->n_local + i;
    printf("vertex %d, parent: %d, level: %d, is_leaf: %d\n",g->ghost_unmap[i], parents[curr], levels[curr],is_leaf[curr]);
  }*/
  
  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing BCC-LCA stage\n");
  }

  
  
  //1. calculate descendants for each owned vertex, counting itself as its own descendant
  uint64_t* n_desc = new uint64_t[g->n_total];
  calculate_descendants(g,comm,q,parents,n_desc);
  std::cout<<"Finished calculating descendants\n";
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
  std::cout<<"Finished calculating preorder labels\n";
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
  
  //3. calculate high and low values for each owned vertex
  uint64_t* lows = new uint64_t[g->n_total];
  uint64_t* highs = new uint64_t[g->n_total];
  calculate_high_low(g, comm, q, highs, lows, parents, preorder);
  std::cout<<"Finished high low calculation\n";
  /*for(int i = 0; i < g->n_local; i++){
    printf("Rank %d: Vertex %d has a high of %d and a low of %d\n",procid,g->local_unmap[i],highs[i],lows[i]);
  }
  for(int i = g->n_local; i < g->n_total; i++){
    printf("Rank %d: Vertex %d has a high of %d and a low of %d\n",procid,g->ghost_unmap[i-g->n_local],highs[i],lows[i]);  
  }*/
  //4. create auxiliary graph.
  dist_graph_t* aux_g = new dist_graph_t;
  std::map< std::pair<uint64_t, uint64_t> , uint64_t > edgeToAuxVert;
  std::map< uint64_t, std::pair<uint64_t, uint64_t> > auxVertToEdge;
  create_aux_graph(g,comm,q,aux_g,edgeToAuxVert,auxVertToEdge,preorder,lows,highs,n_desc,parents);
  std::cout<<"Rank "<<procid<<"Finished creating auxiliary graph:\n";
  
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
  std::cout<<"aux_g->n_total = "<<aux_g->n_total<<"\n";
  std::cout<<"aux_g->n_local = "<<aux_g->n_local<<"\n";
  uint64_t* labels = new uint64_t[aux_g->n_total];
  for(int i = 0; i < aux_g->n_total; i++) labels[i] = 0;
  aux_connectivity_check(aux_g, edgeToAuxVert,auxVertToEdge, labels);
  std::cout<<"Rank "<<procid<<" finished propagating through aux graph\n";
  /*std::cout<<"Rank "<<procid<<"OUTPUTTING AUX GRAPH ********\n";
  for(int i = 0; i < aux_g->n_local; i++){
    std::pair<uint64_t, uint64_t> edge = auxVertToEdge[i];
    std::cout<<"Edge {"<<edge.first<<","<<edge.second<<"} is labeled "<<labels[i]<<"\n";
  }*/
  //6. remap labels to original edges and extend the labels to include certain non-tree edges that were excluded from the auxiliary graph.
  uint64_t* bicc_labels = new uint64_t[g->m_local];
  for(int i = 0; i < g->m_local; i++) bicc_labels[i] = 0;
  finish_edge_labeling(g,aux_g,edgeToAuxVert,auxVertToEdge,preorder,parents,labels,bicc_labels);
  std::cout<<"Rank "<<procid<< " finished final edge labeling\n";
  std::cout<<"aux_g->n = "<<aux_g->n_total<<" aux_g->m = "<<aux_g->m<<"\n";
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
    uint64_t bicc = 0;
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      if(bicc == 0){
        bicc = bicc_labels[g->out_edges[j]];
      } else {
        if(bicc != bicc_labels[g->out_edges[j]]){
          artpts[n_artpts++] = g->local_unmap[i];
          break;
        }
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

  if (verbose) {
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
  delete [] labels;
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


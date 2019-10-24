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

void create_aux_graph(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, dist_graph_t* aux_g, std::map<std::pair<uint64_t, uint64_t>, uint64_t> &edgeToAuxVert,
                      std::map<uint64_t, std::pair<uint64_t, uint64_t> > &auxVertToEdge, uint64_t* preorder, uint64_t* lows, uint64_t* highs, uint64_t* n_desc, uint64_t* parents){
  
  uint64_t n_srcs = 0;
  uint64_t n_dsts = 0;
  std::cout<<"Calculating the number of edges in the aux graph\n";  
  for(uint64_t v = 0; v < g->n_local; v++){
    for(uint64_t j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; j++){
      uint64_t w = g->out_edges[j];
      uint64_t w_global = 0;
      if(w<g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(g->local_unmap[v] == parents[w] || parents[v] == w_global){ //this is a tree edge
        if(preorder[v] == 1) continue;
        if((lows[w] < preorder[v] || highs[w] >= preorder[v] + n_desc[v])||(lows[v] < preorder[w] || highs[v]>=preorder[w])){
          n_srcs+=2;
          n_dsts+=2;
        }
      } else {
        if(preorder[v] + n_desc[v] <= preorder[w] || preorder[w] + n_desc[w] <= preorder[v]){
          n_srcs+=2;
          n_dsts+=2;
        }
      }
    }
  }
  
  uint64_t** srcs = new uint64_t*[n_srcs];
  uint64_t** dsts = new uint64_t*[n_dsts];
  //entries in these arrays take the form {v, w}, where v and w are global vertex identifiers.
  //additionally we ensure v < w.
  for(uint64_t i = 0; i < n_srcs; i++){
    srcs[i] = new uint64_t[2];
    dsts[i] = new uint64_t[2];
  }
  uint64_t srcIdx = 0;
  uint64_t dstIdx = 0;
  std::cout<<"Creating srcs and dsts arrays\n";
  for(uint64_t v = 0; v < g->n_local; v++){
    for(uint64_t j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; j++){
      uint64_t w = g->out_edges[j];//w is local, not preorder
      uint64_t w_global = 0;
      if(w < g->n_local) w_global = g->local_unmap[w];
      else w_global = g->ghost_unmap[w-g->n_local];
      if(g->local_unmap[v] == parents[w] || parents[v] == w_global){
        if(preorder[v] == 1) continue;
        if((lows[w] < preorder[v] || highs[w] >= preorder[v] + n_desc[v])||(lows[v] < preorder[w] ||highs[v] >= preorder[w])){
          //add the sorted edge to the arrays
          //add {parents[v], v} as a src and {v, w} as a dst
	  std::cout<<"Rank "<<procid<<" adding {"<<parents[v]<<","<<g->local_unmap[v]<<"} -- {"<< g->local_unmap[v]<<","<<w_global<<"} as an edge\n";
          if(parents[v] < g->local_unmap[v]){
            srcs[srcIdx][0] = parents[v];
            srcs[srcIdx++][1] = g->local_unmap[v];
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          } else {
            srcs[srcIdx][0] = g->local_unmap[v];
            srcs[srcIdx++][1] = parents[v];
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          }
          if(g->local_unmap[v] < w_global){
            dsts[dstIdx][0] = g->local_unmap[v];
            dsts[dstIdx++][1]=w_global;
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          } else {
            dsts[dstIdx][0] = w_global;
            dsts[dstIdx++][1] = g->local_unmap[v];
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          }
          //add {v, w} as a src and {parents[v], v} as a dst
          if(parents[v] < g->local_unmap[v]){
            dsts[dstIdx][0] = parents[v];
            dsts[dstIdx++][1] = g->local_unmap[v];
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          } else {
            dsts[dstIdx][0] = g->local_unmap[v];
            dsts[dstIdx++][1] = parents[v];
            //std::cout<<"added to dsts[dstIdx][0] = "<<dsts[dstIdx - 1][0]<<" dsts[dstIdx][1] = "<<dsts[dstIdx-1][1]<<"\n";
          }
          if(g->local_unmap[v] < w_global){
            srcs[srcIdx][0] = g->local_unmap[v];
            srcs[srcIdx++][1]=w_global;
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          } else {
            srcs[srcIdx][0] = w_global;
            srcs[srcIdx++][1] = g->local_unmap[v];
            //std::cout<<"added to srcs[srcIdx][0] = "<<srcs[srcIdx - 1][0]<<" srcs[srcIdx][1] = "<<srcs[srcIdx-1][1]<<"\n";
          }
          
        }
      } else {
        if(preorder[v] + n_desc[v] <= preorder[w] || preorder[w] + n_desc[w] <= preorder[v]){
          //add the sorted edge to the arrays
          //add {parents[v], v} as a src and {parents[w], w} as a dst
	  std::cout<<"Rank "<<procid<<" adding {"<<parents[v]<<","<<g->local_unmap[v]<<"} -- {"<<parents[w]<<","<<w_global<<"} as an edge\n";
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
      }
    }
  }
  std::cout<<"Finished populating srcs and dsts arrays:\n";
  
  //for(int i = 0; i < n_srcs; i++){
  //  std::cout<<"("<<srcs[i][0]<<","<<srcs[i][1]<<") -- ("<<dsts[i][0]<<","<<dsts[i][1]<<")\n";
  //}
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
      if(get_value(g->map,auxVert.first) < g->n_local && get_value(g->map,auxVert.second) < g->n_local) {// both are not ghosts (good!)
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
  uint64_t aux_n_local = currId;
  //loop back through the arrays and add the ghosts to the indices  
  for(uint64_t i = 0; i < srcIdx; i++){
    std::pair<uint64_t, uint64_t> auxVert = std::make_pair(srcs[i][0], srcs[i][1]);
    if(edgeToAuxVert.find(auxVert) == edgeToAuxVert.end()){ //the edge doesn't already exist, so it's ghosted
      edgeToAuxVert.insert(std::make_pair(auxVert,currId));
      auxVertToEdge.insert(std::make_pair(currId,auxVert));
      currId++;
    }
    std::cout<<"Edge {"<<auxVert.first<<","<<auxVert.second<<"} mapped to local ID: "<<edgeToAuxVert[auxVert]<<"\n";
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
      if(get_value(g->map,auxVert.first) >= g->n_local){
        std::cout<<"Aux Vertex "<<auxVertId<<" is owned by rank "<<g->ghost_tasks[get_value(g->map,auxVert.first)-g->n_local]<<"\n";
        ghost_tasks[auxVertId-aux_n_local] = g->ghost_tasks[get_value(g->map, auxVert.first) - g->n_local];
      } else if (get_value(g->map,auxVert.second) >= g->n_local){
        std::cout<<"Aux Vertex "<<auxVertId<<" is owned by rank "<<g->ghost_tasks[get_value(g->map,auxVert.second)-g->n_local]<<"\n";
        ghost_tasks[auxVertId-aux_n_local] = g->ghost_tasks[get_value(g->map, auxVert.second) - g->n_local];
      }
      /*if(auxVert.first < g->n_local){
        ghost_tasks[auxVertId] = g->ghost_tasks[get_value(g->map,auxVert.second) - g->n_local];
      } else {
        ghost_tasks[auxVertId] = g->ghost_tasks[get_value(g->map,auxVert.first) - g->n_local];
      }*/
    }
  }

  for(int i =0; i<aux_n_local; i++){
    std::cout<<"aux vertex "<<i<<" has degree "<<aux_degrees[i]<<"\n";
  }
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
  }

  status = MPI_Alltoallv(sendbuf,sendcnts,sdispls,MPI_INT,recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);
  
  for(int exchangeIdx = 0; exchangeIdx < recvsize; exchangeIdx += 3){
    std::pair<uint64_t, uint64_t> edge = std::make_pair(recvbuf[exchangeIdx], recvbuf[exchangeIdx+1]);
    uint64_t vert = edgeToAuxVert[edge];
    uint64_t owned_label = labels[vert];
    uint64_t recvd_label = recvbuf[exchangeIdx+2];
    if(owned_label == 0){
      labels[vert] = recvd_label;
      if(vert < aux_g->n_local){
        int out_degree = out_degree(aux_g, vert);
        uint64_t* outs = out_vertices(aux_g, vert);
        for( int i = 0; i < out_degree; i++){
          frontier.push(outs[i]);
        }
      }
    } else if (recvd_label == 0){
      recvbuf[exchangeIdx + 2] = owned_label;
    } else if (recvd_label != owned_label){
      std::cout<<"a received label doesn't match the owned label, this should not happen\n";
    }
  }

  status = MPI_Alltoallv(recvbuf, recvcnts, rdispls, MPI_INT, sendbuf, sendcnts, sdispls, MPI_INT, MPI_COMM_WORLD);
  //this is back where the verts are ghosted, should not attempt to add to frontier.
  for(int updateIdx = 0; updateIdx < sendsize; updateIdx += 3){
    std::pair<uint64_t, uint64_t> edge = std::make_pair(recvbuf[updateIdx], recvbuf[updateIdx+1]);
    uint64_t vert = edgeToAuxVert[edge];
    uint64_t owned_label = labels[vert];
    uint64_t recvd_label = recvbuf[updateIdx +2];
    if(owned_label == 0){
      labels[vert] = recvd_label;
    } else if (recvd_label != 0 && owned_label != recvd_label){
      std::cout<<"owned and received labels mismatch, this should not happen\n";
    }
  }
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
    
    int propProc = -1;
    if(n_unlabeled > 0) propProc = procid;
    int global_propProc = -1;
    MPI_Allreduce(&propProc, &global_propProc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::cout<<"Rank "<<procid<<" global_propProc = "<<global_propProc<<"\n";
    //the selected process propagates from any unlabeled vertex, creating a frontier as it goes.
    std::queue<uint64_t> frontier;
    if(global_propProc == procid){
      for(int i = 0; i < aux_g->n_local; i++){
        if(labels[i] == 0){
          frontier.push(i);
          break;
        }
      }
    }
    uint64_t currLabel = global_propProc + nprocs*round;
    //while this propagation is not finished, do local work
    while(!global_done){
      while(frontier.size() > 0){
        uint64_t currVert = frontier.front();
        frontier.pop();
        if(labels[currVert] == 0) labels[currVert] = currLabel;
        if(currVert < aux_g->n_local){ //we don't have adjacency info for ghosted verts
          int out_degree = out_degree(aux_g, currVert);
          uint64_t* outs = out_vertices(aux_g, currVert);
          for(int j = 0; j < out_degree; j++){
            if(labels[outs[j]] == 0) {
              std::cout<<"Rank "<<procid<<" adding vertex "<<outs[j]<<" to the frontier\n";
              frontier.push(outs[j]);
            }
          }
        }
      }
      //once the frontier is empty, communicate to all processes using the aux_communicate function
      std::cout<<"Rank "<<procid<<" is starting communication\n";
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
    for(int i = 0; i < aux_g->n_total; i++){
      if(labels[i] == 0) n_unlabeled++;
    }
    MPI_Allreduce(&n_unlabeled, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout<<"number of unlabeled vertices left in the global graph "<<global_done<<"\n";
    //if there are no unlabeled vertices anywhere, the entire process is done.
    global_done = !global_done;
  }
}

void finish_edge_labeling(dist_graph_t* g, dist_graph_t* aux_g, const std::map< std::pair<uint64_t, uint64_t>, uint64_t> & edgeToAuxVert, 
			  const std::map< uint64_t, std::pair<uint64_t, uint64_t> > & auxVertToEdge, uint64_t* preorder, uint64_t* parents,
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
      uint64_t nbor = g->out_vertices[edgeIdx];
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
          for( updateIdx = 0; updateIdx < sendsize; updateIdx+=3){
            std::pair<uint64_t, uint64_t> update_edge;
            update_edge.first = sendbuf[updateIdx];
            update_edge.second = sendbuf[updateIdx+1];
            uint64_t update_label = sendbuf[updateIdx+2];
            if((update_edge.first == nbor_global || update_edge.second == nbor_global) && (update_edge.first == parents[nbor] || update_edge.second == parents[nbor])){
              final_labels[edgeIdx] = update_label;
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
  
}

extern "C" int bicc_dist(dist_graph_t* g,mpi_data_t* comm, queue_data_t* q)
{

  double elt = 0.0, elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

  for(int i = 0; i < g->n_local; i++){
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
  //4. create auxiliary graph.
  dist_graph_t* aux_g = new dist_graph_t;
  std::map< std::pair<uint64_t, uint64_t> , uint64_t > edgeToAuxVert;
  std::map< uint64_t, std::pair<uint64_t, uint64_t> > auxVertToEdge;
  create_aux_graph(g,comm,q,aux_g,edgeToAuxVert,auxVertToEdge,preorder,lows,highs,n_desc,parents);
  std::cout<<"Finished creating auxiliary graph:\n";
  
  for(int i = 0; i < aux_g->n_local; i++){
    std::cout<<"\taux vertex "<<i;//<<" is adjacent to:\n";
    int out_degree = out_degree(aux_g, i);
    std::cout<<" has degree "<<out_degree<<"\n";
    uint64_t* outs = out_vertices(aux_g, i);
    for(int j = 0; j < out_degree; j++){
      std::cout<<"\t\taux vertex "<<outs[j];
      if(outs[j] >= aux_g->n_local){
        std::cout<<" owned by rank "<<aux_g->ghost_tasks[outs[j]-aux_g->n_local];
      }
      std::cout<<"\n";
    }
  }
  //5. flood-fill labels to determine which edges are in the same biconnected component, communicating the labels between copies of the same edge.
  std::cout<<"aux_g->n_total = "<<aux_g->n_total<<"\n";
  std::cout<<"aux_g->n_local = "<<aux_g->n_local<<"\n";
  uint64_t* labels = new uint64_t[aux_g->n_total];
  aux_connectivity_check(aux_g, edgeToAuxVert,auxVertToEdge, labels);
  for(int i = 0; i < aux_g->n_local; i++){
    std::pair<uint64_t, uint64_t> edge = auxVertToEdge[i];
    std::cout<<"Edge {"<<edge.first<<","<<edge.second<<"} is labeled "<<labels[i]<<"\n";
  }
  //6. remap labels to original edges and extend the labels to include certain non-tree edges that were excluded from the auxiliary graph.
  uint64_t* bicc_labels = new uint64_t[g->m_local];
  for(uint64_t i = 0; i < g->n_local; i++){
    int out_degree = out_degree(g, i);
    uint64_t* outs = out_vertice(g, i);
    for(int j = 0; j < out_degree; j++){
      uint64_t nbor = outs[j];
      //if the edge is non-tree
        //if the preorder for i is greater than the preorder for nbor, or vice versa
	  //if the vertex that is greater is locally owned
	    //set the label, the edge we need is locally owned
	  //else
	    //add the global edge we need to some sort of communication set.
    }
  }  

 
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


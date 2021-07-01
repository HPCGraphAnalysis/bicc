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
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <cstring>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <limits>
#include <unordered_map>
#include "bicc_dist.h"

#include "comms.h"
#include "dist_graph.h"
#include "bfs.h"
#include "lca.h"
#include "art_pt_heuristic.h"
#include "label_prop.h"
//#include "color-bicc.h"
//#include "label.h"

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

extern "C" int bicc_dist(dist_graph_t* g,mpi_data_t* comm, queue_data_t* q)
{

  double elt = 0.0;//, elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

 for(int i = 0; i < g->n_local; i++){
    int out_degree = out_degree(g, i);
    uint64_t* outs = out_vertices(g, i);
    printf("Task %d: global %lu (local %lu) neighbors:\n\t",procid,g->local_unmap[i],i);
    for(int j = 0; j < out_degree; j++){
      printf("global %lu (local %lu) ",(outs[j] >= g->n_local?g->ghost_unmap[outs[j]-g->n_local]: g->local_unmap[outs[j]]),outs[j]);
    }
    printf("\n");
  }
  uint64_t* parents = new uint64_t[g->n_total];
  uint64_t* levels = new uint64_t[g->n_total];
  bicc_bfs_pull(g, comm, q, parents, levels, g->max_degree_vert);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  for(int i = 0; i < g->n_local; i++){
    int curr_global = g->local_unmap[i];
    printf("vertex %d, parent: %d, level: %d\n",curr_global, parents[i], levels[i]);
  }
  for(int i = 0; i < g->n_ghost; i++){
    int curr = g->n_local + i;
    printf("vertex %d, parent: %d, level: %d\n",g->ghost_unmap[i], parents[curr], levels[curr]);
  }
  
  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing BCC-LCA stage\n");
  }

  uint64_t* potential_art_pts = new uint64_t[g->n_total];
  
  
  for(uint64_t i = 0; i < g->n_total; i++) potential_art_pts[i] = 0;
  std::cout<<"Doing art_pt_heuristic\n"; 
  art_pt_heuristic(g,comm,q,parents,levels,potential_art_pts);
 
  

  for(uint64_t i = 0; i < g->n_total; i++){
    if(potential_art_pts[i] != 0){
      printf("Task %d: global vertex %lu (local %lu) is a potential articulation point\n",procid,g->local_unmap[i],i);
    }
  }
  
  std::unordered_map<uint64_t, std::set<int>> procs_to_send;
  
  for(uint64_t i = 0; i < g->n_local; i++){
    int degree = out_degree(g, i);
    uint64_t* nbors = out_vertices(g, i);
    for(int j = 0; j < degree; j++){
      if(nbors[j] >= g->n_local){
        procs_to_send[i].insert(g->ghost_tasks[nbors[j] - g->n_local]);
      }
    }
  }
  for(uint64_t i = 0; i < g->n_local; i++){
    printf("Task %d will send global vertex %lu (local %lu) to Task(s): ",procid, g->local_unmap[i], i);
    for(auto it = procs_to_send[i].begin(); it != procs_to_send[i].end(); it++){
      printf("%d ",*it);
    }
    printf("\n");
  }
  
  //communicate potential artpt flags if they are ghosted on a process.
  int* potential_artpt_sendcnts = new int[nprocs];
  int* potential_artpt_recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    potential_artpt_sendcnts[i] = 0;
    potential_artpt_recvcnts[i] = 0;
  }

  for(int i = 0; i < g->n_local; i++){
    if(procs_to_send[i].size() > 0 && potential_art_pts[i] != 0){
      for(auto it = procs_to_send[i].begin(); it != procs_to_send[i].end(); it++){
        potential_artpt_sendcnts[*it] += 1;
      }
    }
  }

  MPI_Alltoall(potential_artpt_sendcnts, 1, MPI_INT, potential_artpt_recvcnts, 1, MPI_INT, MPI_COMM_WORLD);

  int potential_artpt_sendsize = 0;
  int potential_artpt_recvsize = 0;
  int* potential_artpt_sdispls = new int[nprocs+1];
  int* potential_artpt_rdispls = new int[nprocs+1];
  potential_artpt_sdispls[0] = 0;
  potential_artpt_rdispls[0] = 0;
  for(int i = 1; i <= nprocs; i++){
    potential_artpt_sdispls[i] = potential_artpt_sdispls[i-1] + potential_artpt_sendcnts[i-1];
    potential_artpt_rdispls[i] = potential_artpt_rdispls[i-1] + potential_artpt_recvcnts[i-1];
    potential_artpt_sendsize += potential_artpt_sendcnts[i-1];
    potential_artpt_recvsize += potential_artpt_recvcnts[i-1];
  }

  int* potential_artpt_sendbuf = new int[potential_artpt_sendsize];
  int* potential_artpt_recvbuf = new int[potential_artpt_recvsize];
  int* potential_artpt_sendidx = new int[nprocs];
  for(int i = 0; i < nprocs; i++) potential_artpt_sendidx[i] = potential_artpt_sdispls[i];
  for(uint64_t i = 0; i < g->n_local; i++){
    if(procs_to_send[i].size() > 0 && potential_art_pts[i] != 0){
      for(auto it = procs_to_send[i].begin(); it != procs_to_send[i].end(); it++){
        potential_artpt_sendbuf[potential_artpt_sendidx[*it]++] = g->local_unmap[i];
      }
    }
  }
  MPI_Alltoallv(potential_artpt_sendbuf, potential_artpt_sendcnts, potential_artpt_sdispls, MPI_INT,
		potential_artpt_recvbuf, potential_artpt_recvcnts, potential_artpt_rdispls, MPI_INT, MPI_COMM_WORLD);

  for(int i = 0; i < potential_artpt_recvsize; i++){
    potential_art_pts[get_value(g->map, potential_artpt_recvbuf[i])] = 1;
  }


  std::cout<<"Task "<<procid<<": setting up ghost adjacencies -- degrees\n";
  std::cout<<"Task "<<procid<<": n_ghost = "<<g->n_ghost<<"\n";
  //set degree counts for ghosts
  std::vector<uint64_t> ghost_degrees(g->n_ghost, 0);
  uint64_t ghost_adjs_total = 0;
  for(int i = 0; i < g->n_local; i++){
    uint64_t degree = out_degree(g, i);
    uint64_t* nbors = out_vertices(g, i);
    for(int j = 0; j < degree; j++){
      if(nbors[j] >= g->n_local) {
	std::cout<<"Task "<<procid<<": found vertex "<<g->ghost_unmap[nbors[j]-g->n_local]<<"\n";
        ghost_degrees[nbors[j] - g->n_local]++;
      }
    }
  }
  for(int p = 0; p < nprocs; p++){
    if(p == procid){
      for(int i = 0; i < ghost_degrees.size(); i++){
        std::cout<<"Task "<<procid<<": ghost vertex "<<g->ghost_unmap[i]<<" (local "<<i+g->n_local<<") has degree "<<ghost_degrees[i]<<"\n";
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  std::cout<<"Task "<<procid<<": setting up ghost adjacencies -- offsets\n";
  std::vector<uint64_t> ghost_offsets(g->n_ghost+1,0);
  for(int i = 1; i < g->n_ghost+1; i++){
    ghost_offsets[i] = ghost_offsets[i-1] + ghost_degrees[i-1];
    ghost_adjs_total += ghost_degrees[i-1];
  }
  std::vector<uint64_t> ghost_adjs(ghost_adjs_total, 0);
  for(int i = 0; i < ghost_degrees.size(); i++) ghost_degrees[i] = 0;
  
  std::cout<<"Task "<<procid<<": setting up ghost adjacencies -- adjs\n";
  for(int i = 0; i < g->n_local; i++){
    uint64_t degree = out_degree(g, i);
    uint64_t* nbors = out_vertices(g, i);
    for(uint64_t j = 0; j < degree; j++){
      if( nbors[j] >= g->n_local){
        ghost_adjs[ghost_offsets[nbors[j]-g->n_local] + ghost_degrees[nbors[j]-g->n_local]] = i;
	ghost_degrees[nbors[j]-g->n_local]++;
      }
    }
  }
  std::cout<<"Task "<<procid<<": done constructing ghost adjs\n";
  for(int p = 0; p < nprocs; p++){
    if(p == procid){
      for(int i = 0; i < g->n_ghost; i++){
        std::cout<<"Task "<<procid<<": ghost vertex "<<g->ghost_unmap[i]<<" (local "<<i+g->n_local<<") neighbors\n\t";
        for(int j = ghost_offsets[i]; j < ghost_offsets[i+1]; j++){
	  std::cout<<g->local_unmap[ghost_adjs[j]]<<" (local "<<ghost_adjs[j]<<") ";
	}
	std::cout<<"\n";
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  //define the largest possible unsigned int as a sentinel value
  uint64_t max_val = std::numeric_limits<uint64_t>::max();
  //LCA labels are a vector of vectors.
  std::vector<std::set<uint64_t>> LCA_labels(g->n_total,std::set<uint64_t>());
  //low labels are a single int.
  uint64_t* low_labels = new uint64_t[g->n_total];
  for(uint64_t i = 0; i < g->n_total; i++) {
    if(i < g->n_local){
      low_labels[i] = g->local_unmap[i];
    } else {
      low_labels[i] = g->ghost_unmap[i-g->n_local];
    }
  }
 
  int* artpt_flags = new int[g->n_local];  

  bcc_bfs_prop_driver(g, ghost_offsets,ghost_adjs, potential_art_pts, LCA_labels, 
		      low_labels, levels,artpt_flags,
		      procs_to_send);
  
  
  uint64_t* artpts = new uint64_t[g->n_local];
  int n_artpts = 0;
  for(uint64_t i = 0; i < g->n_local; i++){
    if(artpt_flags[i] == 1){
      artpts[n_artpts++] = g->local_unmap[i];
    }
  }

  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) sendcnts[i] = n_artpts;
  int* recvcnts = new int[nprocs];
  MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);

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
  MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);

  if(procid == 0){
    for(int i = 0; i < recvsize; i++){
      std::cout<<"*********Vertex "<<recvbuf[i]<<" is an articulation point*****************\n";
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
  delete [] potential_art_pts;

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


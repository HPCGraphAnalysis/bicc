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

#ifndef _DIST_GRAPH_H_
#define _DIST_GRAPH_H_

#include <stdint.h>

#include "fast_map.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

//struct dist_graph_t;
struct mpi_data_t;
struct queue_data_t;

struct graph_gen_data_t {  
  uint64_t n;
  uint64_t m;
  uint64_t n_local;
  uint64_t n_offset;

  uint64_t m_local_read;
  uint64_t m_local_edges;

  uint64_t *gen_edges;
  uint64_t *global_edge_indices;
};

struct dist_graph_t {
  uint64_t n;
  uint64_t m;
  uint64_t m_local;

  uint64_t n_local;
  uint64_t n_offset;
  uint64_t n_ghost;
  uint64_t n_total;

  uint64_t max_degree_vert;
  uint64_t max_degree;

  uint64_t* out_edges;
  uint64_t* out_degree_list;
  uint64_t* ghost_degrees;

  uint64_t* local_unmap;
  uint64_t* ghost_unmap;
  uint64_t* ghost_tasks;
  uint64_t* n_offsets;
  uint64_t* edge_unmap;
  fast_map* map;
  fast_map* edge_map;
} ;
#define out_degree(g, n) (g->out_degree_list[n+1] - g->out_degree_list[n])
#define out_vertices(g, n) &g->out_edges[g->out_degree_list[n]]


int create_graph(graph_gen_data_t *ggi, dist_graph_t *g);

int create_graph_serial(graph_gen_data_t *ggi, dist_graph_t *g);

int create_graph(dist_graph_t* g, 
          uint64_t n_global, uint64_t m_global, 
          uint64_t n_local, uint64_t m_local,
          uint64_t* local_offsets, uint64_t* local_adjs, uint64_t* global_ids);

int create_graph_serial(dist_graph_t* g, 
          uint64_t n_global, uint64_t m_global, 
          uint64_t n_local, uint64_t m_local,
          uint64_t* local_offsets, uint64_t* local_adjs);

int clear_graph(dist_graph_t *g);

int relabel_edges(dist_graph_t *g);

int relabel_edges(dist_graph_t* g, int32_t* part_list);

int get_max_degree_vert(dist_graph_t *g);

int get_ghost_degrees(dist_graph_t* g);

int get_ghost_degrees(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q);

int repart_graph(dist_graph_t *g, mpi_data_t* comm, int32_t* part_list);

int determine_edge_block(dist_graph_t* g, int32_t*& part_list);

inline int32_t highest_less_than(uint64_t* prefix_sums, uint64_t val)
{
  bool found = false;
  int32_t rank = 0;
  int32_t bound_low = 0;
  int32_t bound_high = nprocs;
  while (!found)
  {
    rank = (bound_high + bound_low) / 2;
    if (prefix_sums[rank] <= val && prefix_sums[rank+1] > val)
    {
      found = true;
    }
    else if (prefix_sums[rank] < val)
      bound_low = rank;
    else if (prefix_sums[rank] > val)
      bound_high = rank;
  }

  return rank;
}

inline int32_t get_rank(dist_graph_t* g, uint64_t vert)
{
  if (vert >= g->n_offsets[procid] && vert < g->n_offsets[procid+1])
    return procid;
  else {
    bool found = false;
    int32_t index = 0;
    int32_t bound_low = 0;
    int32_t bound_high = nprocs;
    while (!found)
    {
      index = (bound_high + bound_low) / 2;
      if (g->n_offsets[index] <= vert && g->n_offsets[index+1] > vert)
      {
        return index;
      }
      else if (g->n_offsets[index] <= vert)
        bound_low = index+1;
      else if (g->n_offsets[index] > vert)
        bound_high = index;
    }

    return index;
  }
  
  return -1;
}

#endif

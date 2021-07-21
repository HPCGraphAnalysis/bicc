/*
//@HEADER
// *****************************************************************************
//
//  HPCGraph: Graph Computation on High Performance Computing Systems
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
#include <time.h>
#include <getopt.h>
#include <cstring>

int procid, nprocs;
bool verbose, debug, debug2, verify, output;

#include "dist_graph.h"
#include "comms.h"
#include "io_pp.h"
#include "bicc.h"
#include "wbter.h"


int main(int argc, char **argv) 
{
  srand(time(0));
  setbuf(stdout, 0);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc < 2) {
    if (procid == 0) printf("To run: %s [graphfile] [root]\n", argv[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  verbose = false;
  debug = false;
  debug2 = false;
  verify = false;
  output = false;
  bool offset_vids = false;

  char* input_prefix = strdup(argv[1]);
  uint64_t root = 0;
  char* part_list = NULL;
  char* nd_filename = NULL;
  char* cd_filename = NULL;
  int64_t* ground_truth = NULL;
  bool wbter = false;

  graph_gen_data_t* ggi = (graph_gen_data_t*)malloc(sizeof(graph_gen_data_t));
  dist_graph_t* g = (dist_graph_t*)malloc(sizeof(dist_graph_t));
  mpi_data_t* comm = (mpi_data_t*)malloc(sizeof(mpi_data_t));

  init_comm_data(comm);
  if (nprocs > 1) {
    if (wbter) {
      generate_bter_MPI(nd_filename, cd_filename, 
        ggi->gen_edges, ground_truth,
        ggi->m_local_read, ggi->n_local, ggi->n_offset, false);
    } else {
      load_graph_edges_32(input_prefix, ggi, offset_vids);
    }
    exchange_edges(ggi, comm);
    create_graph(ggi, g);
    relabel_edges(g);
    if (part_list != NULL)
      repart_graph(g, comm, part_list);
  } else {
    load_graph_edges_32(input_prefix, ggi, offset_vids);
    create_graph_serial(ggi, g);
  }  
  free(ggi);

  get_max_degree_vert(g);
  root = g->max_degree_vert;
  bicc_dist(g, comm, root);

  clear_graph(g);
  clear_comm_data(comm);
  free(g);
  free(comm);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}


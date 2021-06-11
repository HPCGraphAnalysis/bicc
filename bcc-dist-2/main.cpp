
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <sstream>
#include <string.h>

#include "bicc_dist.h"
#include "dist_graph.h"
#include "generate.h"
#include "comms.h"
#include "io_pp.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

void print_usage(char** argv)
{
  printf("To run: %s [graphfile] [options]\n", argv[0]);
  printf("\t Use -h for list of options\n\n");
  exit(0);
}

void print_usage_full(char** argv)
{
  printf("To run: %s [graphfile] [options]\n\n", argv[0]);
  exit(0);
}


int main(int argc, char **argv) 
{
  srand(time(0));
  setbuf(stdout, 0);

  verbose = true;
  debug = true;
  verify = true;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc < 2) 
  {
    if (procid == 0)
      print_usage_full(argv);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  char input_filename[1024]; input_filename[0] = '\0';
  char graphname[1024]; graphname[0] = '\0';  
  //char* graph_name = strdup(argv[1]);
  strcat(input_filename, argv[1]);

  uint64_t num_runs = 1;
  bool output_time = true;

  bool gen_rmat = false;
  bool gen_rand = false;
  bool gen_hd = false;
  uint64_t gen_n = 0;
  uint64_t gen_m_per_n = 16;
  bool offset_vids = false;

  char c;
  while ((c = getopt (argc, argv, "v:e:o:i:m:s:p:dlqtc")) != -1) {
    switch (c) {
      case 'h': print_usage_full(argv); break;
      case 'm': num_runs = strtoul(optarg, NULL, 10); break;
      case 'r': gen_rmat = true; gen_n = strtoul(optarg, NULL, 10); break;
      case 'g': gen_rand = true; gen_n = strtoul(optarg, NULL, 10); break;
      case 's': gen_hd = true; gen_n = strtoul(optarg, NULL, 10); break;
      case 'p': gen_m_per_n = strtoul(optarg, NULL, 10); break;
      case 'd': offset_vids = true; break;
      case 't': output_time = true; break;
      default:
        throw_err("Input argument format error");
    }
  }

  graph_gen_data_t ggi;
  dist_graph_t g;

  mpi_data_t comm;
  init_comm_data(&comm);
  if (gen_rand)
  {
    std::stringstream ss;
    ss << "rand-" << gen_n << "-" << gen_m_per_n;
    strcat(graphname, ss.str().c_str());
    generate_rand_out_edges(&ggi, gen_n, gen_m_per_n, offset_vids);
  }
  else if (gen_rmat)
  {
    std::stringstream ss;
    ss << "rmat-" << gen_n << "-" << gen_m_per_n;
    strcat(graphname, ss.str().c_str());
    generate_rmat_out_edges(&ggi, gen_n, gen_m_per_n, offset_vids);
  }
  else if (gen_hd)
  {
    std::stringstream ss;
    ss << "hd-" << gen_n << "-" << gen_m_per_n;
    strcat(graphname, ss.str().c_str());
    generate_hd_out_edges(&ggi, gen_n, gen_m_per_n, offset_vids);
  }
  else
  {
    if (procid == 0) printf("Reading in graphfile %s\n", input_filename);
    double elt = omp_get_wtime();
    strcat(graphname, input_filename);
    load_graph_edges_32(input_filename, &ggi, offset_vids);
    if (procid == 0) printf("Reading Finished: %9.6lf (s)\n", omp_get_wtime() - elt);
  }

  if (nprocs > 1)
  {
    exchange_edges(&ggi, &comm);
    create_graph(&ggi, &g);
    relabel_edges(&g);
  }
  else
  {
    create_graph_serial(&ggi, &g);
  }
  get_max_degree_vert(&g);
  
  queue_data_t q;
  init_queue_data(&g, &q);

  double total_elt = 0.0;
  for (uint32_t i = 0; i < num_runs; ++i)
  {
    if (procid == 0) printf("BiCC Decomposition\n");
    double elt = omp_get_wtime();
    bicc_dist(&g, &comm, &q);
    total_elt += omp_get_wtime() - elt;
    if (procid == 0) printf("BiCC Time: %9.6lf (s)\n", omp_get_wtime() - elt);
  }
  if (output_time && procid == 0 && num_runs > 1) 
  {
    printf("BiCC Avg. Time: %9.6lf (s)\n", (total_elt / (double)num_runs) );
  }

  clear_graph(&g);
  clear_comm_data(&comm);
  clear_queue_data(&q);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}


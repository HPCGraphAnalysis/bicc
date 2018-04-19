
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <sys/time.h>
#include <vector>
#include <queue>
#include <getopt.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

bool verbose, debug, random_start;

#define MAX_LEVELS 65536

#include "util.h"
#include "graph.h"
#include "bcc_bfs.h"
#include "bcc_color.h"

void print_usage(char** argv)
{
  printf("To run: %s [graphfile] [options]\n", argv[0]);
  printf("\t Use -h for list of options\n\n");
  exit(0);
}

void print_usage_full(char** argv)
{
  printf("To run: %s [graphfile] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-b:\n");
  printf("\t\tRun BFS BCC algorithm\n");
  printf("\t-c:\n");
  printf("\t\tRun Coloring BCC algorithm\n");
  printf("\t-r:\n");
  printf("\t\tUse random root instead of default max-degree vert\n");
  printf("\t-o [output file]:\n");
  printf("\t\tWrite per-edge BCC assignments\n");
  printf("\t-m [#]:\n");
  printf("\t\tIf very high diameter graph (>65536), use this flag to\n");
  printf("\t\tincrease storage size of internal level-dependent structures\n");
  printf("\t-v:\n");
  printf("\t\tEnable verbose timing output\n");
  printf("\t-d:\n");
  printf("\t\tEnable debug output\n");
  exit(0);
}

void read_edge(char* filename,
  int& n, unsigned& m,
  int*& srcs, int*& dsts)
{

  double elt = 0.0;
  if (verbose) {
    printf("Reading in file %s\n", filename);
    elt = timer();
  }

  std::ifstream infile;
  std::string line;
  infile.open(filename);

  getline(infile, line, ' ');
  n = atoi(line.c_str());
  getline(infile, line);
  m = strtoul(line.c_str(), NULL, 10) * 2;

  if (verbose) {
    printf("n %d, m %u\n", n, m/2);
    elt = timer();
  }

  int src, dst;
  unsigned counter = 0;

  srcs = new int[m];
  dsts = new int[m];
  for (unsigned i = 0; i < (m / 2); ++i)
  {
    getline(infile, line, ' ');
    src = atoi(line.c_str());
    getline(infile, line);
    dst = atoi(line.c_str());

    srcs[counter] = src;
    dsts[counter] = dst;
    ++counter;
    srcs[counter] = dst;
    dsts[counter] = src;
    ++counter;
  }
  assert(counter == m);

  infile.close();

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }

  return;
}


void create_csr(int n, unsigned m, 
  int* srcs, int* dsts,
  int*& out_array, unsigned*& out_degree_list,
  int& max_degree_vert, double& avg_out_degree)
{
  double elt = 0.0;
  if (verbose) {
    printf("Creating graph format\n");
    elt = timer();
  }

  out_array = new int[m];
  out_degree_list = new unsigned[n+1];
  unsigned* temp_counts = new unsigned[n];

#pragma omp parallel for
  for (unsigned i = 0; i < m; ++i)
    out_array[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    temp_counts[i] = 0;

  for (unsigned i = 0; i < m; ++i)
    ++temp_counts[srcs[i]];
  for (int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, n*sizeof(int));
  for (unsigned i = 0; i < m; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];
  delete [] temp_counts;

  unsigned max_degree = 0;
  max_degree_vert = -1;
  avg_out_degree = 0.0;
  for (int i = 0; i < n; ++i)
  {
    unsigned degree = out_degree_list[i+1] - out_degree_list[i];
    avg_out_degree += (double)degree;
    if (degree > max_degree)
    {
      max_degree = degree;
      max_degree_vert = i;
    }
  }
  avg_out_degree /= (double)n;
  assert(max_degree_vert >= 0);
  assert(avg_out_degree >= 0.0);

  
  if (verbose) {
    printf("max degree vert: %d, max degree: %d, avg degree %lf\n", max_degree_vert, 
      (out_degree_list[max_degree_vert+1] - out_degree_list[max_degree_vert]),
      avg_out_degree);
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }
}

void output_bcc(graph* g, int* bcc_maps, char* output_file)
{
  double elt = 0.0;
  if (verbose) {
    printf("Outputting mapping as %s\n", output_file);
    elt = timer();
  }

  std::ofstream outfile;
  outfile.open(output_file);

  for (unsigned i = 0; i < g->m; ++i)
    outfile << bcc_maps[i] << std::endl;

  outfile.close();  

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }
}


int main(int argc, char* argv[])
{
  srand(time(0));
  setbuf(stdout, NULL);  

  if (argc < 3)
    print_usage(argv);

  bool run_bfs = false;
  bool run_color = false;

  verbose = false;
  debug = false;
  random_start = false;
  
  bool output = false; 
  char output_file[1024];
  char graph_file[1024];
  strcpy(graph_file, argv[1]);
  int max_levels = MAX_LEVELS;

  char c;
  while ((c = getopt (argc, argv, "hbcm:vdro:")) != -1) {
    switch (c) {
      case 'h': print_usage_full(argv); break;
      case 'b': run_bfs = true; break;
      case 'c': run_color = true;  break;
      case 'm': max_levels = atoi(optarg); break;
      case 'v': verbose = true; break;
      case 'd': debug = true; break;
      case 'r': random_start = true; break;
      case 'o': 
        output = true;
        strcpy(output_file, optarg);
        break;
      default:
        fprintf(stderr, "Input argument format error");
        exit(1);
    }
  }

  printf("Starting BCC decomposition...\n");

  int* srcs;
  int* dsts;
  int n;
  unsigned m;
  int* out_array;
  unsigned* out_degree_list;
  int max_degree_vert;
  double avg_out_degree;

  read_edge(graph_file, n, m, srcs, dsts);
  create_csr(n, m, srcs, dsts, 
    out_array, out_degree_list, max_degree_vert, avg_out_degree);
  graph g = {n, m, out_array, out_degree_list, max_degree_vert, avg_out_degree};
  delete [] srcs;
  delete [] dsts;

  int* bcc_maps = new int[g.m];

  if (run_bfs)
  	bcc_bfs(&g, bcc_maps, max_levels);
  if (run_color)
  	bcc_color(&g, bcc_maps);

  if (output)
  	output_bcc(&g, bcc_maps, output_file);

  delete [] bcc_maps;
  delete [] out_degree_list;
  delete [] out_array;

  return 0;
}

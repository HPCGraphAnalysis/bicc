
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
  FILE* infile = fopen(filename, "r");
  char line[256];

  n = 0;

  unsigned count = 0;
  unsigned cur_size = 1024*1024;
  srcs = (int*)malloc(cur_size*sizeof(int));
  dsts = (int*)malloc(cur_size*sizeof(int));

  while(fgets(line, 256, infile) != NULL) {
    if (line[0] == '%') continue;

    sscanf(line, "%d %d", &srcs[count], &dsts[count]);
    dsts[count+1] = srcs[count];
    srcs[count+1] = dsts[count];

    if (srcs[count] > n)
      n = srcs[count];
    if (dsts[count] > n)
      n = dsts[count];

    count += 2;
    if (count > cur_size) {
      cur_size *= 2;
      srcs = (int*)realloc(srcs, cur_size*sizeof(int));
      dsts = (int*)realloc(dsts, cur_size*sizeof(int));
    }
  }  
  m = count;

  printf("Read: n: %d, m: %u\n", n, m);

  fclose(infile);

  return;
}

void read_bin(char* filename,
 int& num_verts, unsigned& num_edges,
 int*& srcs, int*& dsts)
{
  num_verts = 0;
  double elt = omp_get_wtime();
  printf("Reading %s ", filename);
#pragma omp parallel
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  FILE *infp = fopen(filename, "rb");
  if(infp == NULL) {
    printf("%d - load_graph_edges() unable to open input file", tid);
    exit(0);
  }

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint32_t));

#pragma omp single
{
  num_edges = (unsigned)nedges_global;
  srcs = new int[num_edges*2];
  dsts = new int[num_edges*2];
}

  uint64_t read_offset_start = tid*2*sizeof(uint32_t)*(nedges_global/nthreads);
  uint64_t read_offset_end = (tid+1)*2*sizeof(uint32_t)*(nedges_global/nthreads);

  if (tid == nthreads - 1)
    read_offset_end = 2*sizeof(uint32_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/(2*sizeof(uint32_t));
  uint32_t* edges_read = (uint32_t*)malloc(2*nedges*sizeof(uint32_t));
  if (edges_read == NULL) {
    printf("%d - load_graph_edges(), unable to allocate buffer", tid);
    exit(0);
  }

  fseek(infp, read_offset_start, SEEK_SET);
  fread(edges_read, nedges, 2*sizeof(uint32_t), infp);
  fclose(infp);
  printf(".");

  uint64_t array_offset = (uint64_t)tid*(nedges_global/nthreads);
  uint64_t counter = 0;
  for (uint64_t i = 0; i < nedges; ++i) {
    int src = (int)edges_read[counter++];
    int dst = (int)edges_read[counter++];
    srcs[array_offset+i] = src;
    dsts[array_offset+i] = dst;
    srcs[array_offset+i+num_edges] = dst;
    dsts[array_offset+i+num_edges] = src;
  }
  

  free(edges_read);
  printf(".");

#pragma omp barrier

#pragma omp for reduction(max:num_verts)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (srcs[i] > num_verts)
      num_verts = srcs[i];
#pragma omp for reduction(max:num_verts)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (dsts[i] > num_verts)
      num_verts = dsts[i]; 
           
} // end parallel

  num_edges *= 2;
  num_verts += 1;
  printf(" Done %9.6lf\n", omp_get_wtime() - elt);
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
  //read_bin(graph_file, n, m, srcs, dsts);
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

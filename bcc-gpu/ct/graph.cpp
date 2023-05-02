#include <fstream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "util.h"

extern int verbose;

int read_bin(char* filename,
 int& n, long& m,
 int*& srcs, int*& dsts)
{
  n = 0;
  double elt = omp_get_wtime();
  if (verbose) printf("Reading %s ... ", filename);
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
  m = (long)nedges_global;
  srcs = new int[m];
  dsts = new int[m];
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

  uint64_t array_offset = (uint64_t)tid*(nedges_global/nthreads);
  uint64_t counter = 0;
  for (uint64_t i = 0; i < nedges; ++i) {
    int src = (int)edges_read[counter++];
    int dst = (int)edges_read[counter++];
    srcs[array_offset+i] = src;
    dsts[array_offset+i] = dst;
  }

  free(edges_read);

#pragma omp barrier

#pragma omp for reduction(max:n)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (srcs[i] > n)
      n = srcs[i];
#pragma omp for reduction(max:n)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (dsts[i] > n)
      n = dsts[i]; 
           
} // end parallel

  m *= 2;
  n += 1;
  if (verbose) printf("Done %lf (s)\n", omp_get_wtime() - elt);
  if (verbose) printf("\tn: %d, m: %li\n", n, m);
  
  return 0;
}

int create_csr(int n, long m, int& max_degree, int& max_degree_vert, double& avg_out_degree,
  int* srcs, int* dsts,
  int*& out_adjlist, long*& out_offsets)
{
  double elt = omp_get_wtime();
  if (verbose) printf("Creating CSR ... ");
  
  out_adjlist = new int[m];
  out_offsets = new long[n+1];
  long* temp_counts = new long[n];

#pragma omp parallel for
  for (long i = 0; i < m; ++i)
    out_adjlist[i] = 0;
#pragma omp parallel for
  for (long i = 0; i < n+1; ++i)
    out_offsets[i] = 0;
#pragma omp parallel for
  for (long i = 0; i < n; ++i)
    temp_counts[i] = 0;
#pragma omp parallel for
  for (long i = 0; i < m/2; ++i) {
#pragma omp atomic
    ++temp_counts[srcs[i]];
#pragma omp atomic
    ++temp_counts[dsts[i]];
  }
  parallel_prefixsums(temp_counts, out_offsets+1, n);   
  
#pragma omp parallel for  
  for (long i = 0; i < n; ++i)
    temp_counts[i] = out_offsets[i];
#pragma omp parallel for
  for (long i = 0; i < m/2; ++i) {
    int src = srcs[i];
    int dst = dsts[i];
    long index = -1;
#pragma omp atomic capture
  { index = temp_counts[src]; temp_counts[src]++; }
    out_adjlist[index] = dst;
#pragma omp atomic capture
  { index = temp_counts[dst]; temp_counts[dst]++; }
    out_adjlist[index] = src;
  }  
  
  delete [] temp_counts;

  max_degree = 0;
  max_degree_vert = 0;
  avg_out_degree = 0.0;
#pragma omp parallel for reduction(max:max_degree)
  for (long i = 0; i < n; ++i)
  {
    long degree = out_offsets[i+1] - out_offsets[i];
    avg_out_degree += (double)degree;
    if (degree > max_degree) {
      max_degree = degree;
      max_degree_vert = i;
    }
    
    quicksort(&out_adjlist[out_offsets[i]], 0, degree-1);
  }

  avg_out_degree /= (double)n;
  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
  
  return 0;
}


graph* create_graph(char* filename)
{
  int* srcs;
  int* dsts;
  int n;
  long m;
  int max_degree;
  int max_degree_vert;
  double avg_out_degree;
  int* out_adjlist;
  long* out_offsets;

  read_bin(filename, n, m, srcs, dsts);
  create_csr(n, m, max_degree, max_degree_vert, avg_out_degree, srcs, dsts, out_adjlist, out_offsets);
  delete [] srcs;
  delete [] dsts;

  graph* g = (graph*)malloc(sizeof(graph));
  g->n = n;
  g->m = m;
  g->max_degree = max_degree;
  g->max_degree_vert = max_degree_vert;
  g->avg_out_degree = avg_out_degree;
  g->out_adjlist = out_adjlist;
  g->out_offsets = out_offsets;

  return g;
}

int clear_graph(graph* g)
{
  g->n = 0;  
  g->m = 0;
  g->max_degree = 0;
  delete [] g->out_adjlist;
  delete [] g->out_offsets;

  return 0;
}

int write_graph(char* filename, 
  long new_num_edges, int* new_srcs, int* new_dsts)
{
  FILE* outfile = fopen(filename, "wb");
  
  uint32_t edge[2];
  for (long i = 0; i < new_num_edges; ++i) {
    edge[0] = (uint32_t)new_srcs[i];
    edge[1] = (uint32_t)new_dsts[i];
    fwrite(edge, sizeof(uint32_t), 2, outfile);
  }
  
  fclose(outfile);
 
  /*
  outfile = fopen("tmp", "w");  
  
  for (int i = 0; i < new_num_edges; ++i) 
    fprintf(outfile, "%d %d\n", new_srcs[i], new_dsts[i]);

  fclose(outfile);
  */

  
  return 0; 
}
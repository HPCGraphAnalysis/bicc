
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "io.h"
#include "graph.h"

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

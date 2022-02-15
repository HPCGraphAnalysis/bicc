
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

#include "fast_ts_map.cpp"

typedef struct {
  int n;
  unsigned m;
  int* out_array;
  unsigned* out_degree_list;
} graph;
#define out_degree(g, n) (g.out_degree_list[n+1] - g.out_degree_list[n])
#define out_vertices(g, n) (&g.out_array[g.out_degree_list[n]])

bool no_multiedges = true;

void read_ebin(char* filename,
 int& num_verts, unsigned& num_edges,
 int*& srcs, int*& dsts)
{
  double elt = omp_get_wtime();
  printf("Begin read_ebin()\n");
  
  num_verts = 0;
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
  srcs = new int[num_edges];
  dsts = new int[num_edges];
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
  printf("Done read_ebin(): %lf (s)\n", omp_get_wtime() - elt); 
  printf("Read: n: %d, m: %u\n", num_verts, num_edges);
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
    if (count >= cur_size) {
      cur_size *= 2;
      srcs = (int*)realloc(srcs, cur_size*sizeof(int));
      dsts = (int*)realloc(dsts, cur_size*sizeof(int));
    }
  }  
  m = count;
  ++n;

  printf("Read: n: %d, m: %u\n", n, m);

  fclose(infile);

  return;
}

void create_csr(int n, unsigned m, 
  int* srcs, int* dsts,
  int*& out_array, unsigned*& out_degree_list)
{
  printf("Building csr ... \n");
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

  for (unsigned i = 0; i < m / 2; ++i) {
    ++temp_counts[srcs[i]];
    ++temp_counts[dsts[i]];
  }
  for (int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, n*sizeof(int));
  for (unsigned i = 0; i < m / 2; ++i) {
    out_array[temp_counts[srcs[i]]++] = dsts[i];
    out_array[temp_counts[dsts[i]]++] = srcs[i];
  }
  delete [] temp_counts;
  printf("Done\n");
}

void get_largest_comp(graph g, int& new_n, int& new_m, 
  int*& new_srcs, int*& new_dsts)
{
  printf("Extracting comp ... \n");
  bool* visited = new bool[g.n];
  for (int i = 0; i < g.n; ++i)
    visited[i] = false;
  int* queue = new int[g.n];
  int* next_queue = new int[g.n];
  int queue_size = 1;
  int next_size = 0;
  queue[0] = 0;
  visited[0] = true;
  
  while (queue_size > 0) {
    printf("%d\n", queue_size);
    for (int i = 0; i < queue_size; ++i) {
      int v = queue[i];
      int* N = out_vertices(g, v);
      for (int j = 0; j < out_degree(g, v); ++j) {
        int u = N[j];
        if (!visited[u]) {
          visited[u] = true;
          next_queue[next_size++] = u;
        }
      }
    }
    
    int* temp = queue;
    queue = next_queue;
    next_queue = temp;
    queue_size = next_size;
    next_size = 0;
  }
  
  new_n = 0;
  new_m = 0;
  int* map = new int[g.n];
  for (int i = 0; i < g.n; ++i)
    map[i] = -1;
  
  for (int i = 0; i < g.n; ++i)
    if (visited[i])
      map[i] = new_n++;


  fast_ts_map ts_map;
  init_map(&ts_map, g.m*2);

  new_srcs = new int[g.m];
  new_dsts = new int[g.m];
  for (int i = 0; i < g.n; ++i) {
    if (visited[i]) {
      int* N = out_vertices(g, i);
      for (int j = 0; j < out_degree(g, i); ++j) {
        if (map[i] < map[N[j]]) {
          long idx = test_set_value(&ts_map, 
            (unsigned)map[i], (unsigned)map[N[j]], 0);
          if (idx >= 0 && no_multiedges) 
            continue;
          else {
            new_srcs[new_m] = map[i];
            new_dsts[new_m] = map[N[j]];
            ++new_m;
          }
        }
      }
    }
  }
  printf("Done\n");
}

void write_edgelist(char* filename, int n, int m, int* srcs, int* dsts)
{
  FILE* outfile = fopen(filename, "w");  
  
  for (int i = 0; i < m; ++i) 
    fprintf(outfile, "%d %d\n", srcs[i], dsts[i]);

  fclose(outfile);

  return;
}

void write_ebin(char* filename, int n, int m, int* srcs, int* dsts)
{
  printf("Writing to %s ... \n", filename);
  
  FILE* outfile = fopen(filename, "wb");  
  
  for (int i = 0; i < m; ++i) {
    uint32_t edge[2];
    edge[0] = srcs[i];
    edge[1] = dsts[i];
    fwrite(edge, sizeof(uint32_t), 2, outfile);
  }

  fclose(outfile);

  printf("done\n");
  return;
}

int main(int argc, char* argv[])
{
  int n;
  unsigned m;
  int* srcs;
  int* dsts;
  int* out_array;
  unsigned* out_degree_list;
  
  //read_edge(argv[1], n, m, srcs, dsts);
  read_ebin(argv[1], n, m, srcs, dsts);
  create_csr(n, m, srcs, dsts, out_array, out_degree_list);
  graph g = {n, m, out_array, out_degree_list};
  
  int new_n;
  int new_m;
  int* new_srcs;
  int* new_dsts;
  get_largest_comp(g, new_n, new_m, new_srcs, new_dsts);
  //write_edgelist(argv[2], new_n, new_m, new_srcs, new_dsts);
  write_ebin(argv[2], new_n, new_m, new_srcs, new_dsts);
  
  return 0;
}

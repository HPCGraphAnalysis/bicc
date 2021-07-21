#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <assert.h>

struct graph {
  int num_verts;
  long num_edges;
  long max_degree;
  int* out_adjlist;
  long* out_offsets;
} ;

inline int out_degree(graph* g, int v) 
{ 
  return g->out_offsets[v+1] - g->out_offsets[v];
}

inline int* out_neighbors(graph* g, int v) 
{ 
  return &g->out_adjlist[g->out_offsets[v]];
}

void parallel_prefixsums(
  long* in_array, long* out_array, long size)
{
  long* thread_sums;

#pragma omp parallel
{
  long nthreads = (long)omp_get_num_threads();
  long tid = (long)omp_get_thread_num();
#pragma omp single
{
  thread_sums = new long[nthreads+1];
  thread_sums[0] = 0;
}

  long my_sum = 0;
#pragma omp for schedule(static)
  for(long i = 0; i < size; ++i) {
    my_sum += in_array[i];
    out_array[i] = my_sum;
  }

  thread_sums[tid+1] = my_sum;
#pragma omp barrier

  long my_offset = 0;
  for(long i = 0; i < (tid+1); ++i)
    my_offset += thread_sums[i];

#pragma omp for schedule(static)
  for(long i = 0; i < size; ++i)
    out_array[i] += my_offset;
}

  delete [] thread_sums;
}

void read_bin(char* filename,
 int& num_verts, long& num_edges,
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
  num_edges = (long)nedges_global;
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
  printf(" Done %9.6lf\n", omp_get_wtime() - elt);
}


int create_csr(int num_verts, long num_edges, long& max_degree,
  int* srcs, int* dsts,
  int*& out_adjlist, long*& out_offsets)
{
  double elt = omp_get_wtime();
  printf("Creating graph ");

  out_adjlist = new int[num_edges];
  out_offsets = new long[num_verts+1];
  long* temp_counts = new long[num_verts];

#pragma omp parallel for
  for (long i = 0; i < num_edges; ++i)
    out_adjlist[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_verts+1; ++i)
    out_offsets[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = 0;

#pragma omp parallel for
  for (long i = 0; i < num_edges/2; ++i) {
#pragma omp atomic
    ++temp_counts[srcs[i]];
#pragma omp atomic
    ++temp_counts[dsts[i]];
  }
  parallel_prefixsums(temp_counts, out_offsets+1, num_verts);
  for (int i = 0; i < num_verts; ++i)
    assert(out_offsets[i+1] == out_offsets[i] + temp_counts[i]);
#pragma omp parallel for  
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = out_offsets[i];
#pragma omp parallel for
  for (long i = 0; i < num_edges/2; ++i) {
    long index = -1;
    int src = srcs[i];
#pragma omp atomic capture
  { index = temp_counts[src]; temp_counts[src]++; }
    out_adjlist[index] = dsts[i];
    int dst = dsts[i];
#pragma omp atomic capture
  { index = temp_counts[dst]; temp_counts[dst]++; }
    out_adjlist[index] = srcs[i];
  }

  max_degree = 0;
#pragma omp parallel for reduction(max:max_degree)
  for (int i = 0; i < num_verts; ++i)
    if (out_offsets[i+1] - out_offsets[i] > max_degree)
      max_degree = out_offsets[i+1] - out_offsets[i];

  delete [] temp_counts;

  printf(" Done : %9.6lf\n", omp_get_wtime() - elt);
  printf("Graph: n=%d, m=%li, davg=%li, dmax=%li\n", 
    num_verts, num_edges/2, num_edges / num_verts / 2, max_degree);

  return 0;
}

int extract_comp(graph* g, long& new_num_edges, int* new_srcs, int* new_dsts)
{
  bool* visited = new bool[g->num_verts];
  int* queue = new int[g->num_verts];
  int* queue_next = new int[g->num_verts];
  int queue_size = 0;
  int queue_size_next = 0;
  
  for (int i = 0; i < g->num_verts; ++i)
    visited[i] = false;
  
  int root = 0;
  queue[root] = 0;
  visited[root] = true;
  queue_size = 1;
  
  while (queue_size > 0) {
    for (int i = 0; i < queue_size; ++i) {
      int v = queue[i];
      int degree = out_degree(g, v);
      int* adjs = out_neighbors(g, v);
      for (int j = 0; j < degree; ++j) {
        int u = adjs[j];
        if (!visited[u]) {
          visited[u]= true;
          queue_next[queue_size_next++] = u;
        }
      }
    }
    
    int* tmp = queue;
    queue = queue_next;
    queue_next = tmp;
    queue_size = queue_size_next;
    queue_size_next = 0;
  }
  
  for (int v = 0; v < g->num_verts; ++v) {
    if (visited[v]) {
      int degree = out_degree(g, v);
      int* adjs = out_neighbors(g, v);
      for (int j = 0; j < degree; ++j) {
        int u = adjs[j];
        if (v < u) {
          new_srcs[new_num_edges] = v;
          new_dsts[new_num_edges++] = u;
        }
      }
    }
  }
  
  int new_num_verts = 0;
  std::unordered_map<int, int> map;
  for (long i = 0; i < new_num_edges; ++i) {
    if (map.find(new_srcs[i]) == map.end())
      map[new_srcs[i]] = new_num_verts++;
    if (map.find(new_dsts[i]) == map.end())
      map[new_dsts[i]] = new_num_verts++;
    
    new_srcs[i] = map[new_srcs[i]];
    new_dsts[i] = map[new_dsts[i]];
  }
  
  printf("New graph: %d vertices, %li edges\n", new_num_verts, new_num_edges);
  
  return 0;
}

int write_graph(char* filename, long new_num_edges, int* new_srcs, int* new_dsts)
{
  FILE* outfile = fopen(filename, "wb");
  
  uint32_t edge[2];
  for (long i = 0; i < new_num_edges; ++i) {
    edge[0] = (uint32_t)new_srcs[i];
    edge[1] = (uint32_t)new_dsts[i];
    fwrite(edge, sizeof(uint32_t), 2, outfile);
  }
  
  fclose(outfile);
 
  outfile = fopen("tmp", "w");  
  
  for (int i = 0; i < new_num_edges; ++i) 
    fprintf(outfile, "%d %d\n", new_srcs[i], new_dsts[i]);

  fclose(outfile);

  
  return 0; 
}


int main(int argc, char** argv)
{ 
  int num_verts = 0;
  long num_edges = 0;
  int* srcs = NULL;
  int* dsts = NULL;
  
  read_bin(argv[1], num_verts, num_edges, srcs, dsts);
  
  long max_degree = 0;
  int* out_adjlist = NULL;
  long* out_offsets = NULL;
  create_csr(num_verts, num_edges, max_degree, 
    srcs, dsts, out_adjlist, out_offsets);
  delete [] srcs;
  delete [] dsts;
  
  graph g = {num_verts, num_edges, max_degree, out_adjlist, out_offsets};
  
  long new_num_edges = 0;
  int* new_srcs = new int[num_edges];
  int* new_dsts = new int[num_edges];
  extract_comp(&g, new_num_edges, new_srcs, new_dsts);  
  write_graph(argv[2], new_num_edges, new_srcs, new_dsts);
  delete [] new_srcs;
  delete [] new_dsts;
  
  return 0;  
}


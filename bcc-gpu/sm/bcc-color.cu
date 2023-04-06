#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "bcc-color.h"
#include "graph.h"

extern int verbose;
extern int debug;

//**************************************************************************************************
// Helper Functions
//**************************************************************************************************

template <typename T> void swap(T* &p, T* &q) {
  T* tmp = p;
  p = q;
  q = tmp;
}

__global__
void update_queue_next(graph* g, int* queue_next, int* queue_next_size, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < g->n) {
    if (in_queue_next[index]) {
      int queue_index = atomicAdd(queue_next_size, 1);
      queue_next[queue_index] = index;
    }
  }
}

//**************************************************************************************************
// BFS
//**************************************************************************************************

__global__
void bfs_init(graph* g, int* root, int* parents, int* levels, int* queue, int* queue_size)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    if (index == *root) {
      parents[index] = index;
      levels[index] = 0;

      int queue_index = atomicAdd(queue_size, 1);
      queue[queue_index] = index;
    }
    else {
      parents[index] = -1;
      levels[index] = -1;
    }
  }
  
  return;
}

__global__
void bfs_level(graph* g, int* parents, int* levels, int level, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < *queue_size) {
    int vert = queue[index];
    in_queue[vert] = false;

    int degree = out_degree(g, vert);
    int* outs = out_adjs(g, vert);

    for (int j = 0; j < degree; ++j) {
      int out = outs[j];

      if (parents[out] == -1) {
        parents[out] = vert;
        levels[out] = level;
        
        in_queue_next[out] = true;
      }
    }
  }
}

void bcc_bfs(graph* g, int* root, int* parents, int* levels, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Running Spanning Tree BFS ... ");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;
  int init_queue_size = 0;

  bfs_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, root, parents, levels, queue, queue_size);

  cudaDeviceSynchronize();

  int level = 1;
  while (*queue_size) {    
    bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, levels, level, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue_next);
    cudaDeviceSynchronize();

    swap(queue, queue_next);
    swap(in_queue, in_queue_next);

    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    level++;

    cudaDeviceSynchronize();
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// Mutual Parents
//**************************************************************************************************

__device__
int mutual_parent(graph* g, int* levels, int* parents, int vert, int out)
{
  int p_out = parents[out];
  int p_vert = parents[vert];

  if (p_out == vert)
    return vert;
  if (p_vert == out)
    return out;

  if(levels[p_out] < levels[p_vert])
    p_vert = parents[p_vert];
  else if(levels[p_out] > levels[p_vert])
    p_out = parents[p_out];

  while (p_vert != p_out)
  {
    p_vert = parents[p_vert];
    p_out = parents[p_out];
  }

  return p_vert;
}

__global__
void init_mutual_parent(graph* g, int* parents, int* levels, int* high, int* low, int* changes)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < g->n) {
    int v = index;

    low[v] = v;
    high[v] = v;

    int vert_parent = parents[v];
    int max_level = levels[vert_parent];

    high[v] = vert_parent;

    int degree = out_degree(g, v);
    int* outs = out_adjs(g, v);
    for (int i = 0; i < degree; ++i) {
      int u = outs[i];

      if (parents[u] == v || vert_parent == u) continue;

      int mp = mutual_parent(g, levels, parents, v, u);
      int mp_level = levels[mp];

      if (mp_level < max_level) {
        max_level = mp_level;
        high[v] = mp;

        atomicAdd(changes, 1);
      }
    }
  }
}

void bcc_find_mutual_parent(graph* g, int* parents, int* levels, int* high, int* low) 
{
  double elt = omp_get_wtime();

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;
  
  int* changes = NULL;
  int init_changes = 0;
  assert(cudaMallocManaged(&changes, sizeof(graph)) == cudaSuccess);
  cudaMemcpy(&changes, &init_changes, sizeof(int), cudaMemcpyHostToDevice);

  init_mutual_parent<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, levels, high, low, changes);

  cudaDeviceSynchronize();

  if (debug) printf("Init mutual changes: %d\n", *changes);

  if (verbose) printf("Running Mutual Parents ...... ");
  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// Color
//**************************************************************************************************

__global__
void color_init(graph* g, int* queue, int* queue_size)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    int queue_index = atomicAdd(queue_size, 1);
    queue[queue_index] = index;
  }
  
  return;
}

__global__
void color_level(graph* g, int* parents, int* levels, int* high, int* low, 
                 int* queue, int* queue_size, int* queue_next, int* queue_next_size, 
                 bool* in_queue, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < *queue_size) {
    int vert = queue[index];
    in_queue[vert] = false;
    int vert_high = high[vert];
    if (vert_high == vert) return;

    int vert_high_level = levels[vert_high];
    int vert_level = levels[vert];
    int vert_low = low[vert];

    int out_degree = out_degree(g, vert);
    int* outs = out_adjs(g, vert);
    for (int j = 0; j < out_degree; ++j) {
        int out = outs[j];
        int out_high = high[out];
        int out_high_level = levels[out_high];
        if ((parents[out] == vert || parents[vert] == out) && out_high_level == vert_level) continue;

        if (vert_high_level < out_high_level) {
            in_queue_next[out] = true;
            high[out] = vert_high;
            out_high = vert_high;
            in_queue_next[vert] = true;
        }
        if (vert_high == out_high) {
            int out_low = low[out];
            if (vert_low < out_low) {
                in_queue_next[out] = true;
                low[out] = vert_low; // < out_low ? vert_low : out_low;
                in_queue_next[vert] = true;
            } else if (out_low < vert_low && vert_high != out) {
                low[vert] = out_low;
                in_queue_next[vert] = true;
            }
        }
    }
  }
}

void bcc_color(graph* g, int* root, int* parents, int* levels, int* high, int* low,
               int* queue, int* queue_size, int* queue_next, int* queue_next_size, 
               bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();
  if (debug) printf("Starting Coloring Phase ...\n");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;
  int init_queue_size = 0;

  color_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue, queue_size);

  cudaDeviceSynchronize();

  int i = 0;
  while (*queue_size) {    
    if (debug) printf("Iter: %d - queue_size: %d\n", i, *queue_size);

    color_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, levels, high, low, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue_next);
    cudaDeviceSynchronize();

    swap(queue, queue_next);
    swap(in_queue, in_queue_next);

    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    i += 1;

    cudaDeviceSynchronize();
  }

  int global_low = g->n;
  unsigned out_degree = out_degree(g, *root);
  int* outs = out_adjs(g, *root);
  for (unsigned i = 0; i < out_degree; ++i)
    if (global_low > low[outs[i]])
      global_low = low[outs[i]];
  high[*root] = *root;
  low[*root] = global_low;

  if (verbose) printf("Completed Coloring Phase .... ");
  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// Label
//**************************************************************************************************

__global__
void art_point_init(graph* g, bool* art_point)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    art_point[index] = false;
  }
  
  return;
}

__global__
void art_point_set(graph* g, bool* art_point, int* high)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    art_point[high[index]] = true;
  }
  
  return;
}

void art_point_root(graph* g, bool* art_point, int* low, int* root) 
{
  art_point[*root] = false;
  unsigned out_degree = out_degree(g, *root);
  int* outs = out_adjs(g, *root);
  for (unsigned i = 0; i < out_degree; ++i) {
    if (low[*root] != low[outs[i]])
    {
      art_point[*root] = true;
      break;
    }
  }
}

__global__
void art_point_final(graph* g, bool* art_point, int* art_point_count)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    if (art_point[index]) {
      atomicAdd(art_point_count, 1);
    }
  }
  
  return;
}

__global__
void bcc_assigned_init(graph* g, bool* bcc_assigned)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->m) {
    bcc_assigned[index] = false;
  }
  
  return;
}

__global__
void bcc_assigned_final(graph* g, bool* bcc_assigned, int* bcc_count)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->m) {
    if (bcc_assigned[index]) {
      atomicAdd(bcc_count, 1);
    }
  }
  
  return;
}

__global__
void label(graph* g, int* high, int* low, int* bcc_maps, int* bcc_map_counter, bool* bcc_assigned) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < g->n) {
    int v = index;

    unsigned begin = g->out_offsets[v];
    unsigned end = g->out_offsets[v+1];
    for (unsigned i = begin; i < end; ++i)
    {
      int out = g->out_adjlist[i];
      if (high[v] == out)
        bcc_maps[i] = low[v];
      else if (high[out] == v)
        bcc_maps[i] = low[out];
      else if (low[out] == low[v])
        bcc_maps[i] = low[v];
      else {
        int value = atomicAdd(bcc_map_counter, 1);
        bcc_maps[i] = value;
      }

      if (bcc_maps[i] >= 0)
        bcc_assigned[bcc_maps[i]] = true;
    }
  }
}

void bcc_label(graph* g, int* root, int* high, int* low, int* bcc_maps, int* bcc_map_counter, 
               bool* art_point, bool* bcc_assigned, int* art_point_count, int* bcc_count) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Running Final Labelling ..... ");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;
  int thread_blocks_m = g->m / BLOCK_SIZE + 1;

  art_point_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, art_point);
  cudaDeviceSynchronize();

  art_point_set<<<thread_blocks_n, BLOCK_SIZE>>>(g, art_point, high);
  cudaDeviceSynchronize();

  art_point_root(g, art_point, low, root);

  bcc_assigned_init<<<thread_blocks_m, BLOCK_SIZE>>>(g, bcc_assigned);
  cudaDeviceSynchronize();

  label<<<thread_blocks_n, BLOCK_SIZE>>>(g, high, low, bcc_maps, bcc_map_counter, bcc_assigned);
  cudaDeviceSynchronize();

  art_point_final<<<thread_blocks_n, BLOCK_SIZE>>>(g, art_point, art_point_count);
  cudaDeviceSynchronize();

  bcc_assigned_final<<<thread_blocks_m, BLOCK_SIZE>>>(g, bcc_assigned, bcc_count);
  cudaDeviceSynchronize();

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// Main
//**************************************************************************************************

int bcc_color_decomposition(graph* g_h, int* bcc_maps_h, int& bcc_count_h, int& art_point_count_h) {
  double elt = omp_get_wtime();
  if (verbose) printf("Initializing data on GPU .... ");
  
  // copy graph data to GPU
  graph* g = NULL;
  assert(cudaMallocManaged(&g, sizeof(graph)) == cudaSuccess);
  cudaMemcpy(&g->n, &g_h->n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->m, &g_h->m, sizeof(long), cudaMemcpyHostToDevice);

  int num_verts = g_h->n;
  int num_edges = g_h->m;
  assert(cudaMallocManaged(&g->out_adjlist, num_edges*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&g->out_offsets, (num_verts + 1)*sizeof(long)) == cudaSuccess);
  cudaMemcpy(g->out_adjlist, g_h->out_adjlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g->out_offsets, g_h->out_offsets, (num_verts + 1)*sizeof(long), cudaMemcpyHostToDevice);
  
  // init parents & levels array for spanning tree
  int* root = NULL;
  int* parents = NULL;
  int* levels = NULL;
  assert(cudaMallocManaged(&root, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&parents, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&levels, num_verts*sizeof(int)) == cudaSuccess);
  cudaMemcpy(root, &g_h->max_degree_vert, sizeof(int), cudaMemcpyHostToDevice);


  // init high & low array
  int* high = NULL;
  int* low = NULL;
  assert(cudaMallocManaged(&high, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&low, num_verts*sizeof(int)) == cudaSuccess);

  // init bcc variables
  int init_count = 0;
  int* art_point_count = NULL;
  int* bcc_count = NULL;
  int* bcc_maps = NULL;
  int* bcc_map_counter = NULL;
  bool* art_point = NULL;
  bool* bcc_assigned = NULL;
  assert(cudaMallocManaged(&art_point_count, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&bcc_count, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&bcc_maps, num_edges*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&bcc_map_counter, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&art_point, num_verts*sizeof(bool)) == cudaSuccess);
  assert(cudaMallocManaged(&bcc_assigned, num_edges*sizeof(bool)) == cudaSuccess);
  cudaMemcpy(art_point_count, &init_count, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(bcc_count, &init_count, sizeof(int), cudaMemcpyHostToDevice);

  // init queue variables
  int init_queue_size = 0;
  int* queue = NULL;
  int* queue_size = NULL;
  int* queue_next = NULL;
  int* queue_next_size = NULL;
  bool* in_queue = NULL;
  bool* in_queue_next = NULL;
  assert(cudaMallocManaged(&queue, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&queue_size, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&queue_next, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&queue_next_size, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&in_queue, num_verts*sizeof(bool)) == cudaSuccess);
  assert(cudaMallocManaged(&in_queue_next, num_verts*sizeof(bool)) == cudaSuccess);
  cudaMemcpy(queue_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);
    
  // move data to GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(g->out_adjlist, num_edges*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(g->out_offsets, (num_verts + 1)*sizeof(long), device, NULL);
  cudaMemPrefetchAsync(root, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(parents, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(levels, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(high, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(low, num_verts*sizeof(int), device, NULL);
  
  cudaMemPrefetchAsync(art_point_count, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(bcc_count, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(bcc_maps, num_edges*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(bcc_map_counter, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(art_point, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(bcc_assigned, num_edges*sizeof(bool), device, NULL);
  
  cudaMemPrefetchAsync(queue, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue_size, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue_next, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue_next_size, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(in_queue, num_verts*sizeof(bool), device, NULL);
  cudaMemPrefetchAsync(in_queue_next, num_verts*sizeof(bool), device, NULL);

  if (verbose) printf("Done: %lf (s)\n\n", omp_get_wtime() - elt);

  elt = omp_get_wtime();

  // *********************************************************************************
  // run BFS from source = 0 to get spanning tree T = (parents, levels)
  // *********************************************************************************
  bcc_bfs(g, root, parents, levels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // run Mutual Parents / LCA on (G,T) to find high and low labels
  // *********************************************************************************
  bcc_find_mutual_parent(g, parents, levels, high, low);

  // *********************************************************************************
  // run Coloring Phase
  // *********************************************************************************
  bcc_color(g, root, parents, levels, high, low, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // run Final Lablleing Phase
  // *********************************************************************************
  bcc_label(g, root, high, low, bcc_maps, bcc_map_counter, art_point, bcc_assigned, art_point_count, bcc_count);

  if (verbose) printf("BCC-Color Runtime w/o GPU Overhead: %lf (s)\n\n", omp_get_wtime() - elt);

  // *********************************************************************************
  // copy results back to host memory
  // *********************************************************************************
  elt = omp_get_wtime();
  if (verbose) printf("Copying Data from GPU ....... ");

  cudaMemcpy(&art_point_count_h, art_point_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&bcc_count_h, bcc_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(bcc_maps_h, bcc_maps, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
  
  // *********************************************************************************
  // Free cuda memory
  // *********************************************************************************
  cudaFree(g->out_adjlist);
  cudaFree(g->out_offsets);
  cudaFree(g);

  cudaFree(root);
  cudaFree(parents);
  cudaFree(levels);
  cudaFree(high);
  cudaFree(low);

  cudaFree(art_point_count);
  cudaFree(bcc_count);
  cudaFree(bcc_maps);
  cudaFree(art_point);
  cudaFree(bcc_assigned);

  cudaFree(queue);
  cudaFree(queue_size);
  cudaFree(queue_next);
  cudaFree(queue_next_size);
  cudaFree(in_queue);
  cudaFree(in_queue_next);

  return 1;
}

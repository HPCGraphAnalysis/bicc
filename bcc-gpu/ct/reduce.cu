#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "reduce.h"
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

void add_new_edges(int* f_parents, int* t_parents, int* new_srcs, int* new_dsts, int &new_num_edges, int num_verts) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Adding Edges to Output ......... ");

  new_num_edges = 0;
  for (int i = 0; i < num_verts; ++i) {
    int u = f_parents[i];
    int w = t_parents[i];

    if (i != u && u != -1) {
      int src = i < u ? i : u;
      int dst = i < u ? u : i;

      new_srcs[new_num_edges] = src;
      new_dsts[new_num_edges] = dst;
      ++new_num_edges;
    }

    if (w != i && w != u && w != -1) {
      int src = i < w ? i : w;
      int dst = i < w ? w : i;

      new_srcs[new_num_edges] = src;
      new_dsts[new_num_edges] = dst;
      ++new_num_edges;
    }
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
  if (debug) printf("\tnum edges after adding F & T: %d\n", new_num_edges);

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
// BFS - Spanning Tree
//**************************************************************************************************

__global__
void st_bfs_init(graph* g, int* parents, int* queue, int* queue_size)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    // Starting source is default to max_degree -- can change here if desired
    if (index == g->max_degree_vert) {
      parents[index] = index;

      int queue_index = atomicAdd(queue_size, 1);
      queue[queue_index] = index;
    }
    else {
      parents[index] = -1;
    }
  }
  
  return;
}

__global__
void st_bfs_level(graph* g, int* parents, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
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
        
        in_queue_next[out] = true;
      }
    }
  }
}

void spanning_tree(graph* g, int* parents, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Running Spanning Tree BFS ...... ");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;

  st_bfs_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, queue, queue_size);

  cudaDeviceSynchronize();

  int init_queue_size = 0;
  while (*queue_size) {    
    st_bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue_next);
    cudaDeviceSynchronize();

    swap(queue, queue_next);
    swap(in_queue, in_queue_next);

    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// Connected Components
//**************************************************************************************************

__global__
void cc_init(graph* g, int* parents, int* labels, 
             int* queue, int* queue_size, bool* in_queue, bool* in_queue_next)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  // Add all v in G - T to queue
  if (index < g->n){
    labels[index] = index;
    
    queue[index] = index;

    in_queue[index] = true;
    in_queue_next[index] = false;
  }
  
  return;
}

__global__
void cc_level(graph* g, int * parents, int* labels, int* queue, int* queue_size, 
              int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < *queue_size) {
    int v = queue[index];
    in_queue[v] = false;

    int degree = out_degree(g, v);
    int* outs = out_adjs(g, v);

    bool changed = false;
    for (int j = 0; j < degree; ++j) {
      int u = outs[j];

      if (parents[u] == v || parents[v] == u) continue;

      if (labels[v] > labels[u]) {        
        labels[v] = labels[u];
        changed = true;
      }
    }
    
    if (changed) {
      in_queue_next[v] = true;

      for (int j = 0; j < degree; ++j) {
        int u = outs[j];
        if (parents[u] == v || parents[v] == u) continue;
        in_queue_next[u] = true;
      }
    }
  }
}

void connected_components(graph* g, int * parents, int* labels, int* queue, int* queue_size, 
                          int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Running Connected Components ... ");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;

  cc_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, labels, queue, queue_size, in_queue, in_queue_next);
  cudaMemcpy(queue_size, &g->n, sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  int init_queue_size = 0;
  while (*queue_size) {
    cc_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, labels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue_next);
    cudaDeviceSynchronize();

    swap(queue, queue_next);
    swap(in_queue, in_queue_next);

    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// BFS - Spanning Forest
//**************************************************************************************************

__global__
void sf_bfs_level(graph* g, int* t_parents, int* f_parents, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < *queue_size) {
    int vert = queue[index];
    in_queue[vert] = false;

    int degree = out_degree(g, vert);
    int* outs = out_adjs(g, vert);

    for (int j = 0; j < degree; ++j) {
      int out = outs[j];

      if (t_parents[vert] == out || t_parents[out] == vert ||
          f_parents[vert] == out || f_parents[out] == vert)
      continue;

      if (f_parents[out] == -1) {
        f_parents[out] = vert;
        
        in_queue_next[out] = true;
      }
    }
  }
}

__global__
void sf_bfs_init(graph* g, int* labels, int* parents, int* queue, int* queue_size, bool* in_queue, bool* in_queue_next)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    // If label equals itself then it must be a source since it is the lowest label for component
    if (index == labels[index]) {
      parents[index] = index;
      
      in_queue[index] = true;
      in_queue_next[index] = false;

      int queue_index = atomicAdd(queue_size, 1);
      queue[queue_index] = index;
    } else {
      parents[index] = -1;

      in_queue[index] = false;
      in_queue_next[index] = false;
    }
  }
  
  return;
}

void spanning_forest(graph* g, int* labels, int* t_parents, int* f_parents, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;

  sf_bfs_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, labels, f_parents, queue, queue_size, in_queue, in_queue_next);

  cudaDeviceSynchronize();

  if (debug) printf("\tnum cc in G - T: %d\n", *queue_size);
  if (verbose) printf("Running Spanning Forest BFS .... ");

  int init_queue_size = 0;
  while (*queue_size) {    
    sf_bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, t_parents, f_parents, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue_next);
    cudaDeviceSynchronize();

    swap(queue, queue_next);
    swap(in_queue, in_queue_next);

    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

//**************************************************************************************************
// Main
//**************************************************************************************************

int reduce_graph_gpu(graph* g_host, int* new_srcs, int* new_dsts, int& new_num_edges) {
  double elt = omp_get_wtime();
  if (verbose) printf("Initializing data on GPU ....... ");
  
  // copy graph data to GPU
  graph* g = NULL;
  assert(cudaMallocManaged(&g, sizeof(graph)) == cudaSuccess);
  cudaMemcpy(&g->n, &g_host->n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->m, &g_host->m, sizeof(long), cudaMemcpyHostToDevice);

  int num_verts = g_host->n;
  int num_edges = g_host->m;
  assert(cudaMallocManaged(&g->out_adjlist, num_edges*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&g->out_offsets, (num_verts + 1)*sizeof(long)) == cudaSuccess);
  cudaMemcpy(g->out_adjlist, g_host->out_adjlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g->out_offsets, g_host->out_offsets, (num_verts + 1)*sizeof(long), cudaMemcpyHostToDevice);
  
  // init parents array for spanning tree & forest
  int* t_parents = NULL;
  int* f_parents = NULL;
  assert(cudaMallocManaged(&t_parents, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&f_parents, num_verts*sizeof(int)) == cudaSuccess);

  // init connected components array
  int* labels = NULL;
  assert(cudaMallocManaged(&labels, num_verts*sizeof(int)) == cudaSuccess);

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
  
  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
  
  // move data to GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(g->out_adjlist, num_edges*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(g->out_offsets, (num_verts + 1)*sizeof(long), device, NULL);
  cudaMemPrefetchAsync(t_parents, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(f_parents, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(labels, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue_size, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue_next, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(queue_next_size, sizeof(int), device, NULL);
  cudaMemPrefetchAsync(in_queue, num_verts*sizeof(bool), device, NULL);
  cudaMemPrefetchAsync(in_queue_next, num_verts*sizeof(bool), device, NULL);

  // Timer for no-gpu timing data
  elt = omp_get_wtime();

  // *********************************************************************************
  // run initial BFS to get spanning tree T
  // *********************************************************************************
  spanning_tree(g, t_parents, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // Run ConnectedComponents on G - T
  // *********************************************************************************
  connected_components(g, t_parents, labels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // run final BFS from sources in labels to get spanning forest F
  // *********************************************************************************
  spanning_forest(g, labels, t_parents, f_parents, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // Update return values with T & F
  // *********************************************************************************
  add_new_edges(f_parents, t_parents, new_srcs, new_dsts, new_num_edges, num_verts);

  if (verbose) printf("Filtering Runtime w/o GPU Overhead: %lf (s)\n", omp_get_wtime() - elt);

  // *********************************************************************************
  // Free cuda memory
  // *********************************************************************************
  cudaFree(g->out_adjlist);
  cudaFree(g->out_offsets);
  cudaFree(g);

  cudaFree(labels);
  cudaFree(t_parents);
  cudaFree(f_parents);

  cudaFree(queue);
  cudaFree(queue_size);
  cudaFree(queue_next);
  cudaFree(queue_next_size);
  cudaFree(in_queue);
  cudaFree(in_queue_next);

  return 1;
}

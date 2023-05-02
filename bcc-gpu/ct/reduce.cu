#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "reduce.h"
#include "graph.h"

extern int verbose;
extern int debug;

/****************************************************************

##     ## ######## ##       ########  ######## ########   ######  
##     ## ##       ##       ##     ## ##       ##     ## ##    ## 
##     ## ##       ##       ##     ## ##       ##     ## ##       
######### ######   ##       ########  ######   ########   ######  
##     ## ##       ##       ##        ##       ##   ##         ## 
##     ## ##       ##       ##        ##       ##    ##  ##    ## 
##     ## ######## ######## ##        ######## ##     ##  ######  

*****************************************************************/

/*
Templated swap for queues
*/
template <typename T> void swap(T* &p, T* &q) {
  T* tmp = p;
  p = q;
  q = tmp;
}

/*
Adds edges from F and T to the output srcs and dsts
*/
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

    // Only add w if it is not the same edge as u
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

/*
Kernel that adds vertex to the next queue if the in_queue_next flag was set for that vertex
*/
__global__
void update_queue_next(graph* g, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < g->n) {
    in_queue[index] = false;
    if (in_queue_next[index]) {
      int queue_index = atomicAdd(queue_next_size, 1);
      queue_next[queue_index] = index;
    }
  }
}


/********************************************************************************************************************************************

 ######  ########     ###    ##    ## ##    ## #### ##    ##  ######      ######## ########  ######## ########    ########  ########  ######  
##    ## ##     ##   ## ##   ###   ## ###   ##  ##  ###   ## ##    ##        ##    ##     ## ##       ##          ##     ## ##       ##    ## 
##       ##     ##  ##   ##  ####  ## ####  ##  ##  ####  ## ##              ##    ##     ## ##       ##          ##     ## ##       ##       
 ######  ########  ##     ## ## ## ## ## ## ##  ##  ## ## ## ##   ####       ##    ########  ######   ######      ########  ######    ######  
      ## ##        ######### ##  #### ##  ####  ##  ##  #### ##    ##        ##    ##   ##   ##       ##          ##     ## ##             ## 
##    ## ##        ##     ## ##   ### ##   ###  ##  ##   ### ##    ##        ##    ##    ##  ##       ##          ##     ## ##       ##    ## 
 ######  ##        ##     ## ##    ## ##    ## #### ##    ##  ######         ##    ##     ## ######## ########    ########  ##        ###### 

********************************************************************************************************************************************/

/*
Spannning Tree BFS Initialization Kernel
*/
__global__
void st_bfs_init(graph* g, int* root, int* parents, int* levels, int* queue, int* queue_size)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    // Starting source is default to max_degree -- can change here if desired
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

/*
Spanning Tree Top Down BFS Iteration Kernel - Updates all neighbors of only queue vertices
*/
 __global__
 void st_td_bfs_level(graph* g, int* parents, int* levels, int level, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
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

/*
Spanning Tree Bottom Up BFS Iteration Kernel - Updates all vertices with first neighbor in prev level
*/
 __global__
 void st_bu_bfs_level(graph* g, int* parents, int* levels, int level, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
 {
   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 
   if (index < g->n) {
     int vert = index;
     int prev_level = level - 1;
 
     if (levels[vert] < 0) {
       int degree = out_degree(g, vert);
       int* outs = out_adjs(g, vert);
 
       for (int j = 0; j < degree; ++j) {
         int out = outs[j];
 
         if (levels[out] == prev_level) {
           levels[vert] = level;
           parents[vert] = out;
           in_queue_next[vert] = true;
           break;
         }
       }
     }
   }
 }

/*
Hybrid BFS - Generates spanning tree for G, switching between top-down and bottom-up based on next queue parameters
*/
void spanning_tree(graph* g, int* root, int* parents, int* levels, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Running Spanning Tree BFS ...... ");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;
  int init_queue_size = 0;

  bool using_top_down = true;
  bool already_switched = false;
  if (debug) printf("\n\tStarting with top-down BFS\n");

  st_bfs_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, root, parents, levels, queue, queue_size);

  cudaDeviceSynchronize();

  int level = 1;
  while (*queue_size) {    
    // Switch BFS type if necessary
    if (!already_switched) {
      int frontier_size = *queue_size;
      if (using_top_down) {
        double edges_frontier = (double)frontier_size * g->avg_out_degree;
        double edges_remainder = (double)(g->n - frontier_size) * g->avg_out_degree; 
        if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0) {
          if (debug) printf("\tSwitching to bottom-up BFS on level %d\n", level);
          using_top_down = false;
        }
      } else if (((double)g->n / BETA) > frontier_size){
        if (debug) printf("\tSwitching back to top-down BFS on level %d\n", level);
        using_top_down = false;
        already_switched = true;
      }
    }

    // Run iteration of BFS
    if (using_top_down) {
      st_td_bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, levels, level, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
      cudaDeviceSynchronize();
    } else {
      st_bu_bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, levels, level, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
      cudaDeviceSynchronize();
    }

    // Update next queue
    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    // Swap queues
    swap(queue, queue_next);
    swap(in_queue, in_queue_next);
    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    level++;
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

/**********************************************************************************************************************************************************************************

 ######   #######  ##    ## ##    ## ########  ######  ######## ######## ########      ######   #######  ##     ## ########   #######  ##    ## ######## ##    ## ########  ######  
##    ## ##     ## ###   ## ###   ## ##       ##    ##    ##    ##       ##     ##    ##    ## ##     ## ###   ### ##     ## ##     ## ###   ## ##       ###   ##    ##    ##    ## 
##       ##     ## ####  ## ####  ## ##       ##          ##    ##       ##     ##    ##       ##     ## #### #### ##     ## ##     ## ####  ## ##       ####  ##    ##    ##       
##       ##     ## ## ## ## ## ## ## ######   ##          ##    ######   ##     ##    ##       ##     ## ## ### ## ########  ##     ## ## ## ## ######   ## ## ##    ##     ######  
##       ##     ## ##  #### ##  #### ##       ##          ##    ##       ##     ##    ##       ##     ## ##     ## ##        ##     ## ##  #### ##       ##  ####    ##          ## 
##    ## ##     ## ##   ### ##   ### ##       ##    ##    ##    ##       ##     ##    ##    ## ##     ## ##     ## ##        ##     ## ##   ### ##       ##   ###    ##    ##    ## 
 ######   #######  ##    ## ##    ## ########  ######     ##    ######## ########      ######   #######  ##     ## ##         #######  ##    ## ######## ##    ##    ##     ######  

**********************************************************************************************************************************************************************************/

/*
Connected Components Initialization
*/
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

/*
Connected Components Kernel - Updates labels and adds any relevant neighbors to the next queue
*/
__global__
void cc_level(graph* g, int* parents, int* labels, int* queue, int* queue_size, 
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

/*
Connected Components - Generates labels for every vertex
*/
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
    // Run level & update queue next
    cc_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, parents, labels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    // Swap queues
    swap(queue, queue_next);
    swap(in_queue, in_queue_next);

    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

/***************************************************************************************************************************************************************

 ######  ########     ###    ##    ## ##    ## #### ##    ##  ######      ########  #######  ########  ########  ######  ########    ########  ########  ######  
##    ## ##     ##   ## ##   ###   ## ###   ##  ##  ###   ## ##    ##     ##       ##     ## ##     ## ##       ##    ##    ##       ##     ## ##       ##    ## 
##       ##     ##  ##   ##  ####  ## ####  ##  ##  ####  ## ##           ##       ##     ## ##     ## ##       ##          ##       ##     ## ##       ##       
 ######  ########  ##     ## ## ## ## ## ## ##  ##  ## ## ## ##   ####    ######   ##     ## ########  ######    ######     ##       ########  ######    ######  
      ## ##        ######### ##  #### ##  ####  ##  ##  #### ##    ##     ##       ##     ## ##   ##   ##             ##    ##       ##     ## ##             ## 
##    ## ##        ##     ## ##   ### ##   ###  ##  ##   ### ##    ##     ##       ##     ## ##    ##  ##       ##    ##    ##       ##     ## ##       ##    ## 
 ######  ##        ##     ## ##    ## ##    ## #### ##    ##  ######      ##        #######  ##     ## ########  ######     ##       ########  ##        ######  

****************************************************************************************************************************************************************/

/*
Spanning Forest BFS Initialization Kernel
*/
__global__
void sf_bfs_init(graph* g, int* labels, int* parents, int* levels, int* queue, int* queue_size, bool* in_queue, bool* in_queue_next)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (index < g->n) {
    // If label equals itself then it must be a source since it is the lowest label for component
    if (index == labels[index]) {
      parents[index] = index;
      levels[index] = 0;
      
      in_queue[index] = true;
      in_queue_next[index] = false;

      int queue_index = atomicAdd(queue_size, 1);
      queue[queue_index] = index;
    } else {
      parents[index] = -1;
      levels[index] = -1;

      in_queue[index] = false;
      in_queue_next[index] = false;
    }
  }
  
  return;
}

/*
Spanning Forest Top Down BFS Iteration Kernel - Updates all neighbors of only queue vertices
*/
__global__
void sf_td_bfs_level(graph* g, int* t_parents, int* f_parents, int* levels, int level, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < *queue_size) {
    int vert = queue[index];
    in_queue[vert] = false;

    int degree = out_degree(g, vert);
    int* outs = out_adjs(g, vert);

    for (int j = 0; j < degree; ++j) {
      int out = outs[j];

      // Dont use if in T or already has parent
      if (t_parents[vert] == out || t_parents[out] == vert ||
          f_parents[vert] == out || f_parents[out] == vert)
      continue;

      if (f_parents[out] == -1) {
        f_parents[out] = vert;
        levels[out] = level;
        
        in_queue_next[out] = true;
      }
    }
  }
}

/*
Spanning Forest Bottom Up BFS Iteration Kernel - Updates all vertices with first neighbor in prev level
*/
 __global__
 void sf_bu_bfs_level(graph* g, int* t_parents, int* f_parents, int* levels, int level, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
 {
   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 
   if (index < g->n) {
     int vert = index;
     int prev_level = level - 1;
 
     if (levels[vert] < 0) {
       int degree = out_degree(g, vert);
       int* outs = out_adjs(g, vert);
 
       for (int j = 0; j < degree; ++j) {
         int out = outs[j];

         // Dont use if in T or already has parent
        if (t_parents[vert] == out || t_parents[out] == vert ||
            f_parents[vert] == out || f_parents[out] == vert)
          continue;
 
         if (levels[out] == prev_level) {
           levels[vert] = level;
           f_parents[vert] = out;
           in_queue_next[vert] = true;
           break;
         }
       }
     }
   }
 }

 /*
 Hybrid BFS - Generates spanning forest for G-T, with labels as src
 Switches between top-down and bottom-up based on next queue parameters
 */
void spanning_forest(graph* g, int* labels, int* t_parents, int* f_parents, int* levels, int* queue, int* queue_size, int* queue_next, int* queue_next_size, bool* in_queue, bool* in_queue_next) 
{
  double elt = omp_get_wtime();
  if (verbose) printf("Running Spanning Forest BFS .... ");

  int thread_blocks_n = g->n / BLOCK_SIZE + 1;
  int init_queue_size = 0;

  bool using_top_down = true;
  bool already_switched = false;

  sf_bfs_init<<<thread_blocks_n, BLOCK_SIZE>>>(g, labels, f_parents, levels, queue, queue_size, in_queue, in_queue_next);

  cudaDeviceSynchronize();

  if (debug) printf("\n\tNumber of CC in G - T (source count): %d\n", *queue_size);
  if (debug) printf("\tStarting with top-down BFS\n");

  int level = 1;
  while (*queue_size) {   
    // Switch BFS type if necessary
    if (!already_switched) {
      int frontier_size = *queue_size;
      if (using_top_down) {
        double edges_frontier = (double)frontier_size * g->avg_out_degree;
        double edges_remainder = (double)(g->n - frontier_size) * g->avg_out_degree; 
        if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0) {
          if (debug) printf("\tSwitching to bottom-up BFS on level %d\n", level);
          using_top_down = false;
        }
      } else if (((double)g->n / BETA) > frontier_size){
        if (debug) printf("\tSwitching back to top-down BFS on level %d\n", level);
        using_top_down = false;
        already_switched = true;
      }
    }

    // Run iteration of BFS
    if (using_top_down) {
      sf_td_bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, t_parents, f_parents, levels, level, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
      cudaDeviceSynchronize();
    } else {
      sf_bu_bfs_level<<<thread_blocks_n, BLOCK_SIZE>>>(g, t_parents, f_parents, levels, level, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);
      cudaDeviceSynchronize();
    }

    // Update next queue
    update_queue_next<<<thread_blocks_n, BLOCK_SIZE>>>(g, queue_next, queue_next_size, in_queue, in_queue_next);
    cudaDeviceSynchronize();

    // Swap queues
    swap(queue, queue_next);
    swap(in_queue, in_queue_next);
    cudaMemcpy(queue_size, queue_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(queue_next_size, &init_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    level++;
  }

  if (verbose) printf("Done: %lf (s)\n", omp_get_wtime() - elt);
}

/********************************

##     ##    ###    #### ##    ## 
###   ###   ## ##    ##  ###   ## 
#### ####  ##   ##   ##  ####  ## 
## ### ## ##     ##  ##  ## ## ## 
##     ## #########  ##  ##  #### 
##     ## ##     ##  ##  ##   ### 
##     ## ##     ## #### ##    ##

********************************/

int reduce_graph_gpu(graph* g_host, int* new_srcs, int* new_dsts, int& new_num_edges) {
  double elt = omp_get_wtime();
  if (verbose) printf("Initializing data on GPU ....... ");
  
  // copy graph data to GPU
  graph* g = NULL;
  assert(cudaMallocManaged(&g, sizeof(graph)) == cudaSuccess);
  cudaMemcpy(&g->n, &g_host->n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->m, &g_host->m, sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->max_degree, &g_host->max_degree, sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->max_degree_vert, &g_host->max_degree_vert, sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->avg_out_degree, &g_host->avg_out_degree, sizeof(double), cudaMemcpyHostToDevice);

  int num_verts = g_host->n;
  int num_edges = g_host->m;
  assert(cudaMallocManaged(&g->out_adjlist, num_edges*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&g->out_offsets, (num_verts + 1)*sizeof(long)) == cudaSuccess);
  cudaMemcpy(g->out_adjlist, g_host->out_adjlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g->out_offsets, g_host->out_offsets, (num_verts + 1)*sizeof(long), cudaMemcpyHostToDevice);
  
  // init root, parents, levels array for spanning tree & forest
  int* root = NULL;
  int* t_parents = NULL;
  int* t_levels = NULL;
  int* f_parents = NULL;
  int* f_levels = NULL;
  assert(cudaMallocManaged(&root, sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&t_parents, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&t_levels, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&f_parents, num_verts*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&f_levels, num_verts*sizeof(int)) == cudaSuccess);
  cudaMemcpy(root, &g_host->max_degree_vert, sizeof(int), cudaMemcpyHostToDevice);

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
  cudaMemPrefetchAsync(t_levels, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(f_parents, num_verts*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(f_levels, num_verts*sizeof(int), device, NULL);
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
  spanning_tree(g, root, t_parents, t_levels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // Run ConnectedComponents on G - T
  // *********************************************************************************
  connected_components(g, t_parents, labels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // run final BFS from sources in labels to get spanning forest F
  // *********************************************************************************
  spanning_forest(g, labels, t_parents, f_parents, f_levels, queue, queue_size, queue_next, queue_next_size, in_queue, in_queue_next);

  // *********************************************************************************
  // Update return values with T & F
  // *********************************************************************************
  add_new_edges(f_parents, t_parents, new_srcs, new_dsts, new_num_edges, num_verts);

  if (verbose) printf("\nFiltering Runtime W/O GPU Overhead: %lf (s)\n", omp_get_wtime() - elt);

  // *********************************************************************************
  // Free cuda memory
  // *********************************************************************************
  cudaFree(g->out_adjlist);
  cudaFree(g->out_offsets);
  cudaFree(g);

  cudaFree(labels);
  cudaFree(root);
  cudaFree(t_parents);
  cudaFree(t_levels);
  cudaFree(f_parents);
  cudaFree(f_levels);

  cudaFree(queue);
  cudaFree(queue_size);
  cudaFree(queue_next);
  cudaFree(queue_next_size);
  cudaFree(in_queue);
  cudaFree(in_queue_next);

  return 1;
}

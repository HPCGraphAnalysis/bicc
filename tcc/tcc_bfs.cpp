
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "tcc_bfs.h"
#include "graph.h"
#include "thread.h"
#include "fast_ts_map.h"

extern bool verbose, debug, random_start;

#define MAX_LEVELS 65536
#define THREAD_QUEUE_SIZE 1024
#define ALPHA 15.0
#define BETA 24.0

/*
'########:::'#######:::::'########::'########::'######::
 ##.... ##:'##.... ##:::: ##.... ##: ##.....::'##... ##:
 ##:::: ##: ##:::: ##:::: ##:::: ##: ##::::::: ##:::..::
 ##:::: ##: ##:::: ##:::: ########:: ######:::. ######::
 ##:::: ##: ##:::: ##:::: ##.... ##: ##...:::::..... ##:
 ##:::: ##: ##:::: ##:::: ##:::: ##: ##:::::::'##::: ##:
 ########::. #######::::: ########:: ##:::::::. ######::
........::::.......::::::........:::..:::::::::......:::
*/
void tcc_bfs_do_bfs(graph* g, int root,
  int* parents, int* levels, 
  int**& level_queues, int* level_counts, int& num_levels)
{
  int num_verts = g->n;
  double avg_out_degree = g->avg_out_degree;

  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int queue_size = 0;  
  int queue_size_next = 0;

  queue[0] = root;
  queue_size = 1;
  parents[root] = root;
  levels[root] = 0;
  level_queues[0] = new int[1];
  level_queues[0][0] = root;
  level_counts[0] = 1;

  int level = 1;
  double elt, elt2 = 0.0;
  int num_descs = 0;
  int local_num_descs = 0;
  bool use_hybrid = false;
  bool already_switched = false;

#pragma omp parallel
{
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;

  while (queue_size)
  {
    if (debug)
      elt = timer();

    if (!use_hybrid)
    {
  #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
      for (int i = 0; i < queue_size; ++i)
      {
        int vert = queue[i];

        unsigned out_degree = out_degree(g, vert);
        int* outs = out_vertices(g, vert);
        for (unsigned j = 0; j < out_degree; ++j)
        {      
          int out = outs[j];
          if (levels[out] < 0)
          {
            levels[out] = level;
            parents[out] = vert;
            ++local_num_descs;
            add_to_queue(thread_queue, thread_queue_size, 
                         queue_next, queue_size_next, out);
          }
        }
      }
    }
    else
    {
      int prev_level = level - 1;

  #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
      for (int vert = 0; vert < num_verts; ++vert)
      {
        if (levels[vert] < 0)
        {
          unsigned out_degree = out_degree(g, vert);
          int* outs = out_vertices(g, vert);
          for (unsigned j = 0; j < out_degree; ++j)
          {
            int out = outs[j];
            if (levels[out] == prev_level)
            {
              levels[vert] = level;
              parents[vert] = out;
              ++local_num_descs;
              add_to_queue(thread_queue, thread_queue_size, 
                           queue_next, queue_size_next, vert);
              break;
            }
          }
        }
      }
    }
    
    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
#pragma omp barrier

#pragma omp single
{
    if (debug)
      printf("num_descs: %d, local: %d\n", num_descs, local_num_descs);    
    num_descs += local_num_descs;

    if (!use_hybrid)
    {  
      double edges_frontier = (double)local_num_descs * avg_out_degree;
      double edges_remainder = (double)(num_verts - num_descs) * avg_out_degree;
      if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0 && !already_switched)
      {
        if (debug)
          printf("\n=======switching to hybrid\n\n");

        use_hybrid = true;
      }
      if (debug)
        printf("edge_front: %lf, edge_rem: %lf\n", edges_frontier, edges_remainder);
    }
    else
    {
      if ( ((double)num_verts / BETA) > local_num_descs  && !already_switched)
      {
        if (debug)
          printf("\n=======switching back\n\n");

        use_hybrid = false;
        already_switched = true;
      }
    }
    local_num_descs = 0;

    if (debug)
      elt2 = timer();

    level_queues[level] = new int[queue_size_next];
    level_counts[level] = queue_size_next;
    memcpy(level_queues[level], queue_next, queue_size_next*sizeof(int));
    ++num_levels;

    if (debug) {
      elt2 = timer() - elt2;
      printf("create array: %9.6lf\n", elt2);
    }

    queue_size = queue_size_next;
    queue_size_next = 0;
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    ++level;

    if (debug) {
      elt = timer() - elt;
      printf("level: %d  num: %d  time: %9.6lf\n", level-1, queue_size, elt);  
    }
} // end single
#pragma omp barrier
  }
} // end parallel


  if (debug)
    printf("Final num desc: %d\n", num_descs);
  
  delete [] queue;
  delete [] queue_next;
}


void tcc_bfs_find_separators(graph* g, int* parents, int* levels, 
    int** level_queues, int* level_counts, int num_levels,
    int* tcc_labels, int* separators, int& num_separators)
{
  int num_verts = g->n;
  int num_edges = g->m;

  if (debug)
    printf("num levels to explore: %d\n", num_levels);

  bool* visited = new bool[num_verts];
  for (int i = 0; i < num_verts; ++i)
    visited[i] = 0;

  char* count = new char[num_verts];
  for (int i = 0; i < num_verts; ++i)
    count[i] = 0;
  
  int* edges = new int[num_edges*2];
  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int edges_size = 0;
  int queue_size = 0;
  int queue_size_next = 0;
  int next_back = 0;
  int cur_low = 0;
  bool is_cur_art = 0;
  int local_bcc_count = 0;
  int num_tcc = 0;
  
  fast_ts_map map;
  init_map(&map, num_edges*2);
  
  // initally, find all trivial components
  // as graph is biconnected, minimum degree must be at least 2
  for (int i = 0; i < num_verts; ++i) {
    int out_degree = out_degree(g, i);
    if (out_degree == 2) {
      int a = g->out_degree_list[i];
      int u = g->out_array[a];
      int v = g->out_array[a+1];      
      
      printf("found trivial separator: %d %d\n", u, v);
      separators[num_separators*2] = u;
      separators[num_separators*2+1] = v;
      tcc_labels[a] = num_tcc;
      tcc_labels[a+1] = num_tcc;
      printf("labeling %d %d as %d\n", i, u, num_tcc);
      printf("labeling %d %d as %d\n", i, v, num_tcc);
      
      
      for (int j = g->out_degree_list[u]; j < g->out_degree_list[u+1]; ++j) {
        if (g->out_array[j] == i) {
          printf("labeling %d %d as %d\n", u, g->out_array[j], num_tcc);
          tcc_labels[j] = num_tcc;
        }
      }
      for (int j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; ++j) {
        if (g->out_array[j] == i) {
          printf("labeling %d %d as %d\n", v, g->out_array[j], num_tcc);
          tcc_labels[j] = num_tcc;
        }
      }
      ++num_separators;
      ++num_tcc;
    }
  }
  
  for (int i = 0; i < num_levels; ++i) {
    printf("level %d:", i);
    for (int j = 0; j < level_counts[i]; ++j)
      printf(" %d", level_queues[i][j]);
    printf("\n");
  }
      
      
for (int l = num_levels-1; l > 0; --l)
{
  int* cur_queue = level_queues[l];
  int level_size = level_counts[l];

  for (int v = 0; v < level_size; ++v)
  {
    int cur_root = cur_queue[v];
    if (out_degree(g, cur_root) == 2) continue;
    
    int root_level = l;
    int root_parent = parents[cur_root];
    int root_parent_level = levels[root_parent];
    
    printf("\n\n\nroot %d level %d parent %d\n", cur_root, root_level, root_parent);

    queue[0] = cur_root;
    queue_size = 1;
    visited[cur_root] = true;
    visited[root_parent] = true;
    count[cur_root] = 2;
    is_cur_art = false;
    int min_level = levels[root_parent];
    int min_level_count = 0;
    int max_level = 0;
    int max_level_next = levels[cur_root];
    bool found_separator = false;
    edges_size = 0;
    
    while (queue_size) {
      printf("queue_size %d - ", queue_size);
      for (int i = 0; i < queue_size; ++i)
        printf(" %d (%d)", queue[i], levels[queue[i]]);
      printf("\n");
      
      // if (queue_size == 1 && (levels[queue[0]] == min_level || 
      //                         levels[queue[0]] == max_level)) {
      //   // queued vertex and the initial parent are separators
      //   separators[num_separators*2] = root_parent;
      //   separators[num_separators*2+1] = queue[0];
        
      //   printf("separator found: %d %d\n", root_parent, queue[0]);
        
      //   // label all the edges discovered during this traverse
      //   for (int i = 0; i < edges_size; i += 2) {
      //     int u = edges[i];
      //     int v = edges[i+1];
      //     for (int j = g->out_degree_list[u]; j < g->out_degree_list[u+1]; ++u)
      //       if (g->out_array[j] == v)
      //         tcc_labels[edges[i]] = num_tcc;
      //     for (int j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; ++v)
      //       if (g->out_array[j] == u)
      //         tcc_labels[edges[i]] = num_tcc;
      //   }
      //   ++num_tcc;          
        
      //   break;
      // }
      
      for (int i = 0; i < queue_size; ++i) {
        int vert = queue[i];        
        printf("vert %d from queue\n", vert);
        
        // don't traverse to lower level until all higher levels explored
        // if (levels[vert] == min_level && max_level != min_level) {
        //   printf("vert %d pushed to next queue\n", vert);
        //   queue_next[queue_size_next++] = vert;
        //   if (levels[vert] > max_level_next)
        //     max_level_next = levels[vert];
        //   continue;
        // }
        
        int begin = g->out_degree_list[vert];
        int end = g->out_degree_list[vert+1];
        for (int j = begin; j < end; ++j) {
          int out = g->out_array[j];
          ++count[out];
          
          
          printf("out %d level %d with count %d\n", out, levels[out], (int)count[out]);
          
          if (out == root_parent) {
            printf("out %d is root parent\n", out);
            continue;
          }
          
          // if out is higher level than our root parent, root parent isn't
          // a primary separator
          if (levels[out] < root_parent_level) {
            printf("high level, root_parent %d not primary \n", root_parent);
            goto exit_loop;
          }
                  
          // don't go through a prior discovered separator
          // if (tcc_labels[j] >= 0) {
          //   printf("edge %d %d goes through separator\n", vert, out);
          //   continue;
          // } else {    
            printf("adding %d %d to edges\n", vert, out);
            edges[edges_size++] = vert;
            edges[edges_size++] = out;
          // }
          
          // only add to queue if we're discovering for second time
          // also now we can add edges 
          if (count[out] == 2 || vert == cur_root) {
            printf("adding %d to queue\n", out);
            queue_next[queue_size_next++] = out;
            //++count[out];
                        
            // edges[edges_size++] = vert;
            // edges[edges_size++] = out;     
          }
          
          // mark edges even if we have discovered out vertex
          //visited[out] = true;
          
          // if (levels[out] > max_level_next)
          //   max_level_next = levels[out];
          // if (levels[out] < min_level)
          //   min_level = levels[out];
          
          printf("max level %d min level %d\n", max_level_next, min_level);
        }
      }
      int* tmp = queue;
      queue = queue_next;
      queue_next = tmp;
      queue_size = queue_size_next;
      queue_size_next = 0;
      max_level = max_level_next;
      max_level_next = 0;     
    }
    // we're done fully exploring a tcc
    
    printf("done exploring tcc with %d as primary\n", root_parent);
    
    for (int i = 0; i < edges_size; i += 2) {
      int u = edges[i];
      int v = edges[i+1];
      printf("edge %d %d %d %d\n", u, v, count[u], count[v]);
      
      // if count lower than 2, not in this tcc
      // we have identified a new separator though
      if (count[u] < 2) {
        if (!found_separator) {
          printf("found separator %d %d\n", root_parent, v);
          found_separator = true;
          separators[num_separators*2] = root_parent;
          separators[num_separators*2+1] = v;
          ++num_separators;
        }
        continue;
      }
      if (count[v] < 2) {
        if (!found_separator) {
        printf("found separator %d %d\n", root_parent, u);
          found_separator = true;
          separators[num_separators*2] = root_parent;
          separators[num_separators*2+1] = u;
          ++num_separators;
        }
        continue;
      }        
      
      for (int j = g->out_degree_list[u]; j < g->out_degree_list[u+1]; ++j)
        if (g->out_array[j] == v)
          tcc_labels[j] = num_tcc;
      for (int j = g->out_degree_list[v]; j < g->out_degree_list[v+1]; ++j)
        if (g->out_array[j] == u)
          tcc_labels[j] = num_tcc;
    }
    ++num_tcc; 
    
exit_loop:   
    // reset visitation and count status
    // for (int i = 0; i < edges_size; i += 2) {
    //   int u = edges[i];
    //   int v = edges[i+1];
    //   visited[u] = false;
    //   visited[v] = false;
    // }
    // visited[cur_root] = false;
    for (int i = 0; i < num_verts; ++i) {
      visited[i] = 0;
      count[i] = 0;
    }
  }
} // end level

  delete [] visited;
  delete [] edges;
  delete [] queue;
  delete [] queue_next;
}


void tcc_bfs(graph* g)
{
  int num_verts = g->n;
  int num_edges = g->m;
  int* parents = new int[num_verts];
  int* levels = new int[num_verts];
  int* separators = new int[num_edges];
  int* tcc_labels = new int[num_edges];
  int num_separators = 0;

  int num_levels = 0;
  int* level_counts = new int[MAX_LEVELS];
  int** level_queues = new int*[MAX_LEVELS];

#pragma omp parallel
{
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    levels[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    parents[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_edges; ++i)
    separators[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_edges; ++i)
    tcc_labels[i] = -1;
#pragma omp for
  for (int i = 0; i < MAX_LEVELS; ++i)
    level_queues[i] = NULL;
}

  int root = -1;
  if (random_start)
    root = rand() % num_verts;
  else
    root = g->max_degree_vert;

  tcc_bfs_do_bfs(g, root,
    parents, levels, 
    level_queues, level_counts, num_levels);

  tcc_bfs_find_separators(g, parents, levels, 
    level_queues, level_counts, num_levels, 
    tcc_labels, separators, num_separators);


  delete [] parents;
  delete [] levels;
  delete [] separators;
  delete [] tcc_labels;
  delete [] level_counts;
  for (int i = 0; i < MAX_LEVELS; ++i)
    if (level_queues[i] == NULL)
      break;
    else
      delete [] level_queues[i];
  delete [] level_queues;
}


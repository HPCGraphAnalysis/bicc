
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "bcc_bfs.h"
#include "graph.h"
#include "thread.h"

extern bool verbose, debug, random_start;

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
void bcc_bfs_do_bfs(graph* g, int root,
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

  }
} // end parallel


  if (debug)
    printf("Final num desc: %d\n", num_descs);
  
  delete [] queue;
  delete [] queue_next;
}




/*
'########:'####:'##::: ##:'########::::::::'###::::'########::'########:
 ##.....::. ##:: ###:: ##: ##.... ##::::::'## ##::: ##.... ##:... ##..::
 ##:::::::: ##:: ####: ##: ##:::: ##:::::'##:. ##:: ##:::: ##:::: ##::::
 ######:::: ##:: ## ## ##: ##:::: ##::::'##:::. ##: ########::::: ##::::
 ##...::::: ##:: ##. ####: ##:::: ##:::: #########: ##.. ##:::::: ##::::
 ##:::::::: ##:: ##:. ###: ##:::: ##:::: ##.... ##: ##::. ##::::: ##::::
 ##:::::::'####: ##::. ##: ########::::: ##:::: ##: ##:::. ##:::: ##::::
..::::::::....::..::::..::........::::::..:::::..::..:::::..:::::..:::::
*/
void bcc_bfs_find_arts(graph* g, int* parents, int* levels, 
    int** level_queues, int* level_counts, int num_levels, 
    int* high, int* low, bool* art_point)
{
  int num_verts = g->n;
  int out_count = 0;
  int jump_count = 0;
  int no_childs = 0;
  int global_bcc_count = 0;

  if (debug)
    printf("num levels to explore: %d\n", num_levels);

  bool* global_visited = new bool[num_verts];
  for (int i = 0; i < num_verts; ++i)
    global_visited[i] = 0;

#pragma omp parallel
{
  bool* visited = new bool[num_verts];
  for (int i = 0; i < num_verts; ++i)
    visited[i] = 0;

  int* stack = new int[num_verts];
  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int stack_back;
  int back;
  int next_back;
  int cur_low;
  bool is_cur_art;
  int local_bcc_count = 0;

for (int l = num_levels-1; l > 1; --l)
{
  int* cur_queue = level_queues[l];
  int level_size = level_counts[l];
  double elt = 0.0;

  if (debug) {
#pragma omp single
    printf("level: %d, level_size: %d\n", l, level_size);
    elt = timer();
  }

#pragma omp for schedule(guided) reduction(+:out_count, jump_count, no_childs)
  for (int v = 0; v < level_size; ++v)
  {
    int vert = cur_queue[v];
    if (low[vert] > -1)
      continue;

    int vert_level = l;
    int vert_parent = parents[vert];
    ++out_count;

    queue[0] = vert;
    back = 1;
    stack[0] = vert;
    stack_back = 1;
    visited[vert] = true;
    visited[vert_parent] = true;
    is_cur_art = false;
    cur_low = vert;

    while (back)
    {
      next_back = 0;
      for (int j = 0; j < back; ++j)
      {
        int new_vert = queue[j];
        int* new_outs = out_vertices(g, new_vert);
        int new_out_degree = out_degree(g, new_vert);
        for (int n = 0; n < new_out_degree; ++n)
        {
          int new_out = new_outs[n];
          if (visited[new_out])
          {
            continue;
          }
          else if (high[new_out] == new_vert)
          {
            continue;
          }
          else if (levels[new_out] < vert_level)
          { 
            ++jump_count;
            goto next_out;
          }
          else
          {
            visited[new_out] = true;
            queue_next[next_back++] = new_out;
            stack[stack_back++] = new_out;
            if (new_out < cur_low)
              cur_low = new_out;      
          }
        }
      }

      int* tmp = queue_next;
      queue_next = queue;
      queue = tmp;
      back = next_back;
    }
    art_point[vert_parent] = true;
    is_cur_art = true;

next_out:
    
    if (is_cur_art)  
    {
      ++local_bcc_count;
      for (int j = 0; j < stack_back; ++j)
      {
        int s_vert = stack[j];
        visited[s_vert] = false;
        high[s_vert] = vert_parent;
        low[s_vert] = cur_low;
      }
    }
    else 
    {  
      for (int j = 0; j < stack_back; ++j)
      {
        int s_vert = stack[j];
        visited[s_vert] = false;    
      }
    }
    visited[vert_parent] = false;
  }


  if (debug) {
#pragma omp single
    printf("time: %9.6lf\n", timer() - elt);
  }
} // end level

#pragma omp atomic
  global_bcc_count += local_bcc_count;

  delete [] visited;
  delete [] stack;
  delete [] queue;
  delete [] queue_next;
}// end par

  if (debug) { 
    printf("jump_count: %d\n", jump_count);
    printf("out_count: %d\n", out_count);
    printf("no_childs: %d\n", no_childs);
    printf("bccs found: %d\n", global_bcc_count);
  }
}

/*
'########:'####:'##::: ##::::'###::::'##:::::::
 ##.....::. ##:: ###:: ##:::'## ##::: ##:::::::
 ##:::::::: ##:: ####: ##::'##:. ##:: ##:::::::
 ######:::: ##:: ## ## ##:'##:::. ##: ##:::::::
 ##...::::: ##:: ##. ####: #########: ##:::::::
 ##:::::::: ##:: ##:. ###: ##.... ##: ##:::::::
 ##:::::::'####: ##::. ##: ##:::: ##: ########:
..::::::::....::..::::..::..:::::..::........::
*/
void bcc_bfs_do_final_bfs(graph* g, int root,
    int* parents, int* levels, 
    int** level_queues, int* level_counts, int num_levels, 
    int* high, int* low, bool* art_point)
{
  int num_verts = g->n;
  double avg_out_degree = g->avg_out_degree;

  bool* visited = new bool[num_verts];
  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int queue_size = 0;  
  int queue_size_next = 0;
  int* stack = new int[num_verts];
  int stack_back = 0;
  
  double elt, elt2 = 0.0;
  int num_descs = 0;
  int local_num_descs = 0;
  bool use_hybrid = false;
  bool already_switched = false;

  int global_low = num_verts;
  int level_size = level_counts[1];
#pragma omp parallel
{
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    visited[i] = false;

for (int l = 0; l < level_size; ++l)
{
#pragma omp single
{
  visited[root] = true;
  int v = level_queues[1][l];
  global_low = v;
  use_hybrid = false;
  already_switched = false;
  num_descs = 0;
  local_num_descs = 0;

  if (low[v] > -1)
  {
    queue_size = 0;
    stack_back = 0;
  }
  else
  {
    queue[0] = v;
    queue_size = 1;
    high[v] = 0;
    stack[0] = v;
    stack_back = 1;
  }
}

  while (queue_size)
  {
    if (debug)
      elt = timer();

    if (!use_hybrid)
    {
#pragma omp for schedule(guided) reduction(+:local_num_descs) reduction(min:global_low)
      for (int i = 0; i < queue_size; ++i)
      {
        int vert = queue[i];

        unsigned out_degree = out_degree(g, vert);
        int* outs = out_vertices(g, vert);
        for (unsigned j = 0; j < out_degree; ++j)
        {      
          int out = outs[j];  
          if (!visited[out] && low[out] < 0)
          {
            visited[out] = true;
            ++local_num_descs;
            if (out < global_low)
              global_low = out;
            add_to_queue(thread_queue, thread_queue_size, 
                         queue_next, queue_size_next, out);
          }
        }
      }
    }
    else
    {
#pragma omp for schedule(guided) reduction(+:local_num_descs) reduction(min:global_low)
      for (int vert = 0; vert < num_verts; ++vert)
      {
        if (!visited[vert] && low[vert] < 0)
        {
          unsigned out_degree = out_degree(g, vert);
          int* outs = out_vertices(g, vert);
          for (unsigned j = 0; j < out_degree; ++j)
          {
            int out = outs[j];
            if (visited[out] && out != root)
            {
              visited[vert] = true;
              ++local_num_descs;              
              if (vert < global_low)
                global_low = vert;
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

    memcpy(&stack[stack_back], queue_next, queue_size_next*sizeof(int));
    stack_back += queue_size_next;

    if (debug) {
      elt2 = timer() - elt2;
      printf("create array: %9.6lf\n", elt2);
    }

    queue_size = queue_size_next;
    queue_size_next = 0;
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;

    if (debug) {
      elt = timer() - elt;
      printf("num: %d  time: %9.6lf\n", queue_size, elt);  
    }
} // end single
  } // end while(queue_size)

#pragma omp for schedule(static)
  for (int i = 0; i < stack_back; ++i)
  {
    int vert = stack[i];
    low[vert] = global_low;
    high[vert] = root;
    visited[vert] = false;
  }
} // end level
} // end parallel
  
  global_low = num_verts;
  unsigned out_degree = out_degree(g, root);
  int* outs = out_vertices(g, root);
  for (unsigned i = 0; i < out_degree; ++i)
    if (global_low > low[outs[i]])
      global_low = low[outs[i]];
  for (unsigned i = 0; i < out_degree; ++i)
    if (global_low != low[outs[i]])
    {
      art_point[root] = true;
      break;
    }

  high[root] = root;
  low[root] = global_low;

  delete [] visited;
  delete [] queue;
  delete [] queue_next;
  delete [] stack;
}



/*
'##::::::::::'###::::'########::'########:'##:::::::
 ##:::::::::'## ##::: ##.... ##: ##.....:: ##:::::::
 ##::::::::'##:. ##:: ##:::: ##: ##::::::: ##:::::::
 ##:::::::'##:::. ##: ########:: ######::: ##:::::::
 ##::::::: #########: ##.... ##: ##...:::: ##:::::::
 ##::::::: ##.... ##: ##:::: ##: ##::::::: ##:::::::
 ########: ##:::: ##: ########:: ########: ########:
........::..:::::..::........:::........::........::
*/
void bcc_bfs_do_labeling(graph* g, int* bcc_maps, int& num_bcc, int& num_art,
                         bool* art_point, int* high, int* low, 
                         int* levels, int* parents)
{
  num_bcc = 0;
  num_art = 0;
  int num_verts = g->n;
  unsigned num_edges = g->m;
  int* out_array = g->out_array;
  unsigned* out_degree_list = g->out_degree_list;
  bool* bcc_assigned = NULL;
  if (debug) bcc_assigned = new bool[num_edges];
  int bcc_map_counter = num_verts;

#pragma omp parallel 
{
  if (debug) {
#pragma omp for schedule(static)
    for (unsigned v = 0; v < num_edges; ++v)
      bcc_assigned[v] = false;
  }

#pragma omp for schedule(guided)
  for (int v = 0; v < num_verts; ++v)
  {
    unsigned begin = out_degree_list[v];
    unsigned end = out_degree_list[v+1];
    for (unsigned i = begin; i < end; ++i)
    {
      int out = out_array[i];
      if (high[v] == out)
        bcc_maps[i] = low[v];
      else if (high[out] == v)
        bcc_maps[i] = low[out];
      else if (low[out] == low[v])
        bcc_maps[i] = low[v];
      else
#pragma omp atomic capture
        bcc_maps[i] = bcc_map_counter++;

      if (debug) {
        if (bcc_maps[i] >= 0)
          bcc_assigned[bcc_maps[i]] = true;
      } 
    }
  }

  if (debug) {
#pragma omp for schedule(static) reduction(+:num_art)
    for (int i = 0; i < num_verts; ++i)
      if (art_point[i])
        ++num_art;
  }

  if (debug) {
#pragma omp for schedule(static) reduction(+:num_bcc)
    for (unsigned e = 0; e < num_edges; ++e)
      if (bcc_assigned[e])
        ++num_bcc;
  }

} // end parallel

  delete [] bcc_assigned;
}


/*
'########:::'######:::'######:::::::::::'########::'########::'######::
 ##.... ##:'##... ##:'##... ##:::::::::: ##.... ##: ##.....::'##... ##:
 ##:::: ##: ##:::..:: ##:::..::::::::::: ##:::: ##: ##::::::: ##:::..::
 ########:: ##::::::: ##:::::::'#######: ########:: ######:::. ######::
 ##.... ##: ##::::::: ##:::::::........: ##.... ##: ##...:::::..... ##:
 ##:::: ##: ##::: ##: ##::: ##:::::::::: ##:::: ##: ##:::::::'##::: ##:
 ########::. ######::. ######::::::::::: ########:: ##:::::::. ######::
........::::......::::......::::::::::::........:::..:::::::::......:::
*/
void bcc_bfs(graph* g, int* bcc_maps, int max_levels)
{
  double elt = 0.0, elt2 = 0.0;
  elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-BFS init\n");
  }

  int num_verts = g->n;
  int num_bcc = 0;
  int num_art = 0;
  int* parents = new int[num_verts];
  int* levels = new int[num_verts];
  int* high = new int[num_verts];
  int* low = new int[num_verts];
  bool* art_point = new bool[num_verts];

  int num_levels = 0;
  int* level_counts = new int[max_levels];
  int** level_queues = new int*[max_levels];

#pragma omp parallel
{
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    levels[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    parents[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    high[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    low[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    art_point[i] = false;
#pragma omp for
  for (int i = 0; i < max_levels; ++i)
    level_queues[i] = NULL;
}

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);  
    elt = timer();
    printf("Doing BCC-BFS BFS stage\n");
  }

  int root = -1;
  if (random_start)
    root = rand() % num_verts;
  else
    root = g->max_degree_vert;

  bcc_bfs_do_bfs(g, root,
    parents, levels, 
    level_queues, level_counts, num_levels);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing BCC-BFS find articulation point stage\n");
  }

  bcc_bfs_find_arts(g, parents, levels, 
    level_queues, level_counts, num_levels, 
    high, low, art_point);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);  
    elt = timer();
    printf("Doing BCC-BFS final BFS stage\n");
  }

  bcc_bfs_do_final_bfs(g, root,
    parents, levels, 
    level_queues, level_counts, num_levels, 
    high, low, art_point);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt); 
    elt = timer();
    printf("Doing BCC-BFS labeling stage\n"); 
  }

  bcc_bfs_do_labeling(g, bcc_maps, num_bcc, num_art, 
                      art_point, high, low, levels, parents);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }

  elt2 = timer() - elt2;
  printf("BCC-BFS Done: %9.6lf (s)\n\n", elt2);
  if (debug) {
    printf("Number of BCCs found: %d\n", num_bcc);
    printf("Number of articulation vertices found: %d\n", num_art);
    if (art_point[root])
      printf("root is art\n");
    else 
      printf("root is not art\n");
  }

  if (debug) {
    printf("===============\n");
    printf("Performing verification\n");

    bool error = false;
    for (int i = 0; i < num_verts; ++i)
    {
      if (high[i] == -1) {
        printf("Error (high) unassigned: %d\n", i);
        error = true;
      }
      if (low[i] == -1) {
        printf("Error (low) unassigned: %d\n", i);
        error = true;
      }
    }

    if (!error)
      printf("===============\nNo errors found\n\n");
    else
      printf("===============\nErrors found\n\n");
  }

  delete [] parents;
  delete [] levels;
  delete [] high;
  delete [] low;
  delete [] art_point;
  delete [] level_counts;
  for (int i = 0; i < max_levels; ++i)
    if (level_queues[i] == NULL)
      break;
    else
      delete [] level_queues[i];
  delete [] level_queues;
}


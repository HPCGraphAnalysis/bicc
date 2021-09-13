
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "bcc_color.h"
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
void bcc_color_do_bfs(graph* g, int root,
  int* parents, int* levels)
{
  int num_verts = g->n;
  double avg_out_degree = g->avg_out_degree;
  bool already_switched = false;

  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int queue_size = 0;  
  int queue_size_next = 0;

  queue[0] = root;
  queue_size = 1;
  parents[root] = root;
  levels[root] = 0;

  int level = 1;
  double elt = 0.0;
  int num_descs = 0;
  int local_num_descs = 0;
  bool use_hybrid = false;

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
'##::::'##:'##::::'##:'########:'##::::'##::::'###::::'##:::::::
 ###::'###: ##:::: ##:... ##..:: ##:::: ##:::'## ##::: ##:::::::
 ####'####: ##:::: ##:::: ##:::: ##:::: ##::'##:. ##:: ##:::::::
 ## ### ##: ##:::: ##:::: ##:::: ##:::: ##:'##:::. ##: ##:::::::
 ##. #: ##: ##:::: ##:::: ##:::: ##:::: ##: #########: ##:::::::
 ##:.:: ##: ##:::: ##:::: ##:::: ##:::: ##: ##.... ##: ##:::::::
 ##:::: ##:. #######::::: ##::::. #######:: ##:::: ##: ########:
..:::::..:::.......::::::..::::::.......:::..:::::..::........::
*/
int bcc_mutual_parent(graph* g, int* levels, int* parents, int vert, int out)
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


/*
'####:'##::: ##:'####:'########::::'##::::'##:'########::
. ##:: ###:: ##:. ##::... ##..::::: ###::'###: ##.... ##:
: ##:: ####: ##:: ##::::: ##::::::: ####'####: ##:::: ##:
: ##:: ## ## ##:: ##::::: ##::::::: ## ### ##: ########::
: ##:: ##. ####:: ##::::: ##::::::: ##. #: ##: ##.....:::
: ##:: ##:. ###:: ##::::: ##::::::: ##:.:: ##: ##::::::::
'####: ##::. ##:'####:::: ##::::::: ##:::: ##: ##::::::::
....::..::::..::....:::::..::::::::..:::::..::..:::::::::
*/
void bcc_color_init_mutual_parents(graph* g, int* parents, int* levels, 
  int* high, int* low)
{
  int num_verts = g->n;
  int changes = 0;

#pragma omp parallel for schedule(guided) reduction(+:changes)
  for (int vert = 0; vert < num_verts; ++vert)
  {
    int vert_parent = parents[vert];
    high[vert] = vert_parent;
    int max_level = levels[vert_parent];

    int out_degree = out_degree(g, vert);
    int* outs = out_vertices(g, vert);
    for (int j = 0; j < out_degree; ++j)
    {
      int out = outs[j];
      if (parents[out] == vert || vert_parent == out)
        continue;

      int mutual_parent = bcc_mutual_parent(g, levels, parents,  vert, out);
      int mp_level = levels[mutual_parent];

      if (mp_level < max_level)
      {
        max_level = mp_level;
        high[vert] = mutual_parent;
        ++changes;
      }
    }
  }

  if (debug)
  	printf("Init mutual changes: %d\n", changes);
}


/*
:'######:::'#######::'##::::::::'#######::'########::
'##... ##:'##.... ##: ##:::::::'##.... ##: ##.... ##:
 ##:::..:: ##:::: ##: ##::::::: ##:::: ##: ##:::: ##:
 ##::::::: ##:::: ##: ##::::::: ##:::: ##: ########::
 ##::::::: ##:::: ##: ##::::::: ##:::: ##: ##.. ##:::
 ##::: ##: ##:::: ##: ##::::::: ##:::: ##: ##::. ##::
. ######::. #######:: ########:. #######:: ##:::. ##:
:......::::.......:::........:::.......:::..:::::..::
*/
void bcc_color_do_coloring(graph* g, int root,
  int* parents, int* levels,
  int* high, int* low)
{

  double elt = 0.0;

  int num_verts = g->n;
  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int queue_size = num_verts;  
  int queue_size_next = 0;  
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];

  int color_changes = 0;
  int color_changes_total = 0;
  int iter = 1;

#pragma omp parallel
{
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    in_queue[i] = false;
#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

  while (queue_size)
  {
		if (debug)
      elt = timer();

#pragma omp for schedule(guided) reduction(+:color_changes) 
    for (int i = 0; i < queue_size; ++i)
    {
      int vert = queue[i];
      in_queue[vert] = false;
      int vert_high = high[vert];
      if (vert_high == vert)
        continue;

      int vert_high_level = levels[vert_high];
      int vert_level = levels[vert];
      int vert_low = low[vert];
      bool vert_change = false;

      int out_degree = out_degree(g, vert);
      int* outs = out_vertices(g, vert);
      for (int j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int out_high = high[out];
        int out_high_level = levels[out_high];
        if (parents[out] == vert && out_high_level == vert_level)
          continue;

        if (vert_high_level < out_high_level)
        {
          if (!in_queue_next[out])
          {
            in_queue_next[out] = true;
            add_to_queue(thread_queue, thread_queue_size, 
                         queue_next, queue_size_next, out);
          }
          high[out] = vert_high;
          out_high = vert_high;
          vert_change = true;
          ++color_changes;   
        }
        if (vert_high == out_high)
        {
          int out_low = low[out];
          if (vert_low < out_low)
          {
            if (!in_queue_next[out])
            {
              in_queue_next[out] = true;
            	add_to_queue(thread_queue, thread_queue_size, 
                           queue_next, queue_size_next, out);
            }
            low[out] = vert_low;// < out_low ? vert_low : out_low;
            vert_change = true;
            ++color_changes;
          }
          else if (out_low < vert_low && vert_high != out)
        	{
        		low[vert] = out_low;
        		vert_change = true;
        		++color_changes;
      		}
        }
      }
      if (vert_change && !in_queue_next[vert])
      {
        in_queue_next[vert] = true;
        add_to_queue(thread_queue, thread_queue_size, 
                     queue_next, queue_size_next, vert);
      }
    }

    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
#pragma omp barrier

#pragma omp single
{        
	  queue_size = queue_size_next;
    queue_size_next = 0;
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;

    color_changes_total += color_changes;
		if (debug) {
		  elt = timer() - elt;
		  printf("Iter %d time %9.6lf\n", iter, elt);
	    printf("\tCurrent changes: %d\n", color_changes);
	    printf("\tTotal changes: %d\n", color_changes_total);
	    printf("\tqueue_size: %d\n", queue_size);
    }
    color_changes = 0;

    ++iter;
}// end single
  } // end while (queue_size)
} // end parallel  

	if (debug) {
		printf("Total iter: %d\n", iter-1);
		printf("Total changes: %d\n", color_changes_total);
	}

  int global_low = num_verts;
  unsigned out_degree = out_degree(g, root);
  int* outs = out_vertices(g, root);
  for (unsigned i = 0; i < out_degree; ++i)
    if (global_low > low[outs[i]])
      global_low = low[outs[i]];
  high[root] = root;
  low[root] = global_low;


  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;
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
void bcc_color_do_labeling(graph* g, int root, 
	int* bcc_maps, int& num_bcc, int& num_art,
  int* high, int* low, 
  int* levels, int* parents)
{
  num_bcc = 0;
  num_art = 0;
  int num_verts = g->n;
  unsigned num_edges = g->m;
  int* out_array = g->out_array;
  unsigned* out_degree_list = g->out_degree_list;
  bool* bcc_assigned = NULL;
  bool* art_point = NULL;
  if (debug) bcc_assigned = new bool[num_edges];  
  if (debug) art_point = new bool[num_verts];
  int bcc_map_counter = num_verts;

#pragma omp parallel 
{
  if (debug) {
#pragma omp for
	  for (int i = 0; i < num_verts; ++i)
	    art_point[i] = false;
#pragma omp for
	  for (int i = 0; i < num_verts; ++i)
	    art_point[high[i]] = true;

#pragma omp single
{  
  art_point[root] = false;
  unsigned out_degree = out_degree(g, root);
  int* outs = out_vertices(g, root);
  for (unsigned i = 0; i < out_degree; ++i)
    if (low[root] != low[outs[i]])
    {
      art_point[root] = true;
      break;
    }
} // end single
  }

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

  if (debug) {
  	delete [] bcc_assigned;
  	delete [] art_point;
	}
}


/*
'########:::'######:::'######::::::::::::'######:::'#######::'##:::::::
 ##.... ##:'##... ##:'##... ##::::::::::'##... ##:'##.... ##: ##:::::::
 ##:::: ##: ##:::..:: ##:::..::::::::::: ##:::..:: ##:::: ##: ##:::::::
 ########:: ##::::::: ##:::::::'#######: ##::::::: ##:::: ##: ##:::::::
 ##.... ##: ##::::::: ##:::::::........: ##::::::: ##:::: ##: ##:::::::
 ##:::: ##: ##::: ##: ##::: ##:::::::::: ##::: ##: ##:::: ##: ##:::::::
 ########::. ######::. ######:::::::::::. ######::. #######:: ########:
........::::......::::......:::::::::::::......::::.......:::........::
*/
void bcc_color(graph* g, int* bcc_maps)
{
  double elt = 0.0, elt2 = 0.0;
  elt2 = timer();
  if (verbose) {
    elt = timer();
    printf("Doing BCC-Color init\n");
  }

  int num_verts = g->n;
  int num_bcc = 0;
  int num_art = 0;
  int* parents = new int[num_verts];
  int* levels = new int[num_verts];
  int* high = new int[num_verts];
  int* low = new int[num_verts];

#pragma omp parallel
{
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    parents[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    levels[i] = -1;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    high[i] = i;
#pragma omp for nowait
  for (int i = 0; i < num_verts; ++i)
    low[i] = i;
}

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);  
    elt = timer();
    printf("Doing BCC-Color BFS stage\n");
  }

  int root = -1;
  if (random_start)
    root = rand() % num_verts;
  else
    root = g->max_degree_vert;

  bcc_color_do_bfs(g, root, parents, levels);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
    elt = timer();
    printf("Doing BCC-Color find mutual parents stage\n");
  }

  bcc_color_init_mutual_parents(g, parents, levels, high, low);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);  
    elt = timer();
    printf("Doing BCC-Color coloring stage\n");
  }
  
  bcc_color_do_coloring(g, root, parents, levels, high, low);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt); 
    elt = timer();
    printf("Doing BCC-Color labeling stage\n"); 
  }

  bcc_color_do_labeling(g, root, bcc_maps, num_bcc, num_art, 
                        high, low, levels, parents);

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }

  elt2 = timer() - elt2;
  printf("BCC-Color Done: %9.6lf (s)\n\n", elt2);
  if (debug) {
    printf("Number of BCCs found: %d\n", num_bcc);
    printf("Number of articulation vertices found: %d\n", num_art);
  }

  delete [] parents;
  delete [] levels;
  delete [] high;
  delete [] low;

}
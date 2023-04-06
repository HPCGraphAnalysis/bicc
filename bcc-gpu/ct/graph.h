#ifndef _GRAPH_H_
#define _GRAPH_H_

struct graph {
  int n;
  long m;
  long max_degree;
  long max_degree_vert;
  int* out_adjlist;
  long* out_offsets;
} ;
#define out_degree(g, n) (g->out_offsets[n+1] - g->out_offsets[n])
#define out_adjs(g, n) &g->out_adjlist[g->out_offsets[n]]

int create_csr(int n, long m, int& max_degree, int& max_degree_vert,
  int* srcs, int* dsts,
  int*& out_adjlist, long*& out_offsets);

graph* create_graph(char* filename);

int clear_graph(graph* g);

int write_graph(char* filename, long new_num_edges, int* new_srcs, int* new_dsts);

#endif

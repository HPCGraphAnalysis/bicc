using namespace std;

#include <getopt.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 

#include "graph.h"
#include "io.h"
#include "reduce.h"

long* global_map;
bool verbose = false;
bool debug = false;

void print_usage(char** argv) 
{
  printf("To run: %s [input ebin graphfile] [output ebin graphfile]\n", argv[0]);
  printf("Options:\n");
  printf("\t-v:\n");
  printf("\t\tEnable verbose timing output\n");
  printf("\t-d:\n");
  printf("\t\tEnable debug output\n");
  exit(0);
}

void check_flags(int argc, char** argv)
{
  char c;
  while ((c = getopt (argc, argv, "vd:")) != -1) {
    switch (c) {
      case 'v': verbose = true; break;
      case 'd': debug = true; break;
      default:
        fprintf(stderr, "Input argument format error");
        exit(1);
    }
  } 
}

int main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  srand(time(0));
  
  if (argc < 3) {
    print_usage(argv);
    exit(0);
  } 

  char input_file[1024];
  char output_file[1024];
  strcpy(input_file, argv[1]);
  strcpy(output_file, argv[2]);

  check_flags(argc, argv);

  graph* g = create_graph(input_file); 

  int initial_edges = g->m;
  int new_num_edges = 0;
  int* new_srcs = new int[g->n*12];
  int* new_dsts = new int[g->n*12];

  printf("\nStarting Edge Filtering on GPU...\n");
  double elt = omp_get_wtime();

  reduce_graph_gpu(g, new_srcs, new_dsts, new_num_edges);
  
  write_graph(output_file, new_num_edges, new_srcs, new_dsts);

  printf("Completed Edge Filtering on GPU: %lf (s)\n", omp_get_wtime() - elt);
  printf("Reduced edge count from %d to %d.\n", initial_edges, new_num_edges * 2);

  clear_graph(g);
  free(new_srcs);
  free(new_dsts);

  return 0;
}

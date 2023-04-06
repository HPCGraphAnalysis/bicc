using namespace std;

#include <getopt.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 

#include "graph.h"
#include "bcc-color.h"
#include "util.h"

#define MAX_LEVELS 65536

bool verbose = false;
bool debug = false;
bool output = false;

void print_usage(char** argv) 
{
  printf("To run: %s [input ebin graphfile]\n", argv[0]);
  printf("Options:\n");
  printf("\t-v:\n");
  printf("\t\tEnable verbose timing output\n");
  printf("\t-d:\n");
  printf("\t\tEnable debug output\n");
  printf("\t-o [output file]:\n");
  printf("\t\tOutput BCC mapping to a file\n");
  exit(0);
}

void check_flags(int argc, char** argv, char* output_file)
{
  char c;
  while ((c = getopt (argc, argv, "hvdo:")) != -1) {
    switch (c) {
      case 'h': print_usage(argv); break;
      case 'v': verbose = true; break;
      case 'd': debug = true; break;
      case 'o': 
        output = true;
        strcpy(output_file, optarg);
        break;
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
  
  if (argc < 2) {
    print_usage(argv);
    exit(0);
  } 

  char input_file[1024];
  char output_file[1024];
  strcpy(input_file, argv[1]);

  check_flags(argc, argv, output_file);

  graph* g = create_graph(input_file); 

  int bcc_count = 0;
  int art_point_count = 0;
  int* bcc_maps = new int[g->m];

  printf("\nStarting Color-BiCC Decomposition on GPU.\n");
  double elt = omp_get_wtime();

  bcc_color_decomposition(g, bcc_maps, bcc_count, art_point_count);

  if (output)
    output_bcc(g, bcc_maps, output_file);
  
  printf("Completed Color-BiCC Decomposition on GPU: %lf (s)\n\n", omp_get_wtime() - elt);
  printf("Number of BCCs found: %d\n", bcc_count);
  printf("Number of articulation vertices found: %d\n", art_point_count);

  clear_graph(g);
  free(bcc_maps);

  return 0;
}

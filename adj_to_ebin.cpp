using namespace std;

#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <vector>
#include <queue>
#include <getopt.h>
#include <string.h>
#include <omp.h>


typedef unsigned char uint8;


typedef struct graph {
  int n;
  long m;
  int* out_array;
  long* out_degree_list;
} graph;

#define out_degree(g, n) (g->out_degree_list[n+1] - g->out_degree_list[n])
#define out_vertices(g, n) &g->out_array[g->out_degree_list[n]]

#include "util.cpp"


void read_adj(char* filename, int& n, long& m,
  int*& out_array, long*& out_degree_list)
{
  ifstream infile;
  string line;
  string val;
  infile.open(filename);

  getline(infile, line, ' ');
  n = atoi(line.c_str());
  //getline(infile, line, ' ');
  getline(infile, line);
  m = atol(line.c_str())*2;
  printf("n: %d, m: %li\n", n, m);
  //getline(infile, line);

  out_array = (int*)malloc(m*sizeof(int));
  out_degree_list = (long*)malloc((n+1)*sizeof(long));

#pragma omp parallel for
  for (long i = 0; i < m; ++i)
    out_array[i] = 0;

#pragma omp parallel for
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;

  long count = 0;
  int cur_vert = 0;

  while (getline(infile, line))
  {
    out_degree_list[cur_vert] = count;
    stringstream ss(line);
    while (getline(ss, val, ' '))
    {
      out_array[count++] = atoi(val.c_str())-1;
    }
    ++cur_vert;
  }
  out_degree_list[cur_vert] = count;

  infile.close();
}

void write_bin(char* filename, graph* g)
{
  FILE *fp;
  fp = fopen(filename, "wb");

  long bad_count = 0;
  for (long i = 0; i < g->n; ++i)
  {
    int out_degree = out_degree(g, i);
    int* outs = out_vertices(g, i);
    for (int j = 0; j < out_degree; ++j)
      if (outs[j] > i) {
        fwrite(&i, sizeof(unsigned), 1, fp);
        fwrite(&outs[j], sizeof(unsigned), 1, fp);
      }
  }

  fclose(fp);
}


int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("\nUsage: %s [infile] [outfile]\n", argv[0]);
    exit(0);
  }
  srand(time(0));

  int n;
  long m;

  double elt = timer();
  printf("reading in graph\n");

  int* out_array;
  long* out_degree_list;
  read_adj(argv[1], n, m, out_array, out_degree_list);
  struct graph g = {n, m, out_array, out_degree_list};
  
  for(int i = 0; i < g.n; i++){
    printf("vertex %d's out edges:\n\t",i);
    int out_degree = out_degree_list[i+1] - out_degree_list[i];
    for(int j = out_degree_list[i]; j < out_degree_list[i+1]; j++){
      printf("%d\n\t",out_array[j]);
    }
  }

  elt = timer() - elt;
  printf("done: %9.6lf\n", elt);
  
  printf("writing\n");
  elt = timer();
  write_bin(argv[2], &g);
  elt = timer() - elt;
  printf("done: %9.2lf\n", elt);

  free(out_array);
  free(out_degree_list);

  return 0;
}

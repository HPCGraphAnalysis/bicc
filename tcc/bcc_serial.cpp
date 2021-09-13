
using namespace std;

#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <sys/time.h>
#include <vector>
#include <queue>
#include <getopt.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif


#define PARALLEL_CUTOFF 10000
#define VERBOSE 1
#define VERIFY 1

#include "util.hpp"
#include "graph.hpp"
#include "bcc_ht.hpp"


void read_in_graph(string filename, 
  int& n, unsigned int& m, 
  int*& srcs, int*& dsts)
{
#if VERBOSE
  printf("Reading in file %s\n", filename.c_str());
    double elt = timer();
#endif

  ifstream file_m;
  string line;  
  file_m.open(filename.c_str());

  // Read in the number of vertices and edges for template and graph
  getline(file_m, line, ' ');
  n = atoi(line.c_str());
  getline(file_m, line);
  m = atoi(line.c_str());

#if VERBOSE
  printf("n: %d  m: %u\n", n, m);
#endif

  if (!n || !m)
  {
    printf("Graph file formatting error\n");
    exit(1);
  }
  
  int src, dst;
  unsigned int counter = 0;
  srcs = new int[m];
  dsts = new int[m];
  for (int i = 0; i < m; ++i)
  {
    getline(file_m, line, ' ');   
    src = atoi(line.c_str());
    getline(file_m, line);
    dst = atoi(line.c_str());

    srcs[counter] = src;
    dsts[counter] = dst;
    ++counter;
  } 
  
  file_m.close();

#if VERBOSE
  elt = timer() - elt;
    printf("Read time: %9.6lf\n", elt);
#endif

  return;
}

int main(int argc, char* argv[])
{
  setbuf(stdout, NULL); 
  if (argc < 2)
  {
    printf("%s [graph in]\n", argv[0]);
    exit(0);
  }

  // Read in and initialize the graph
#if VERBOSE 
    double elt = timer(); 
#endif

  char* graphfile = strdup(argv[1]);
    
  int n;
  unsigned int m;
  int* srcs;
  int* dsts;  
  graph g;
  read_in_graph(graphfile, n, m, srcs, dsts); 
  g.init(n, m, srcs, dsts);

  delete [] srcs;
  delete [] dsts;

#if VERBOSE
    elt = timer() - elt;
    printf("Graph create time: %9.6lf\n", elt);
#endif  
  
  int num_runs = 6;
  int num_threads [] = {1, 2, 4, 8, 16, 32};


  double start = timer();
  biconnected bc(g);
  start = timer() - start;
  printf("Serial time: %9.6lf\n", start);

#if VERIFY
  printf("Art count: %d\n", bc.get_articulation_count(g));
#endif

  g.clear();
  return 0;
}

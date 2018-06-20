#include <stdlib.h>
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

bool verbose, debug, random_start;

#define MAX_LEVELS 65536

#include "util.h"
#include "graph.h"
#include "bcc_bfs.h"
#include "bcc_color.h"

void print_usage(char** argv)
{
  printf("To run: %s [graphfile] [options]\n", argv[0]);
  printf("\t Use -h for list of options\n\n");
  exit(0);
}

void print_usage_full(char** argv)
{
  printf("To run: %s [graphfile] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-b:\n");
  printf("\t\tRun BFS BCC algorithm\n");
  printf("\t-c:\n");
  printf("\t\tRun Coloring BCC algorithm\n");
  printf("\t-r:\n");
  printf("\t\tUse random root instead of default max-degree vert\n");
  printf("\t-o [output file]:\n");
  printf("\t\tWrite per-edge BCC assignments\n");
  printf("\t-m [#]:\n");
  printf("\t\tIf very high diameter graph (>65536), use this flag to\n");
  printf("\t\tincrease storage size of internal level-dependent structures\n");
  printf("\t-v:\n");
  printf("\t\tEnable verbose timing output\n");
  printf("\t-d:\n");
  printf("\t\tEnable debug output\n");
  exit(0);
}

void read_edge(char* filename,
  int& n, unsigned& m,
  int*& srcs, int*& dsts)
{

  double elt = 0.0;
  if (verbose) {
    printf("Reading in file %s\n", filename);
    elt = timer();
  }

  std::ifstream infile;
  std::string line;
  infile.open(filename);

  getline(infile, line, ' ');
  n = atoi(line.c_str());
  getline(infile, line);
  m = strtoul(line.c_str(), NULL, 10) * 2;

  if (verbose) {
    printf("n %d, m %u\n", n, m/2);
    elt = timer();
  }

  int src, dst;
  unsigned counter = 0;

  srcs = new int[m];
  dsts = new int[m];
  for (unsigned i = 0; i < (m / 2); ++i)
  {
    getline(infile, line, ' ');
    src = atoi(line.c_str());
    getline(infile, line);
    dst = atoi(line.c_str());

    srcs[counter] = src;
    dsts[counter] = dst;
    ++counter;
    srcs[counter] = dst;
    dsts[counter] = src;
    ++counter;
  }
  assert(counter == m);

  infile.close();

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }

  return;
}

void read_edge_mesh(char* filename, int &n, unsigned &m, int &z, int*& srcs, int *&dsts, bool *&boundary){
  double elt = 0.0;
  if (verbose) {
    printf("Reading in file %s\n", filename);
    elt = timer();
  }

  std::ifstream infile;
  std::string line;
  infile.open(filename);

  // ignore first line
  getline(infile,line);
  

  getline(infile,line);
  int x = atoi(line.c_str());
  line = line.substr(line.find(" "));
  int y = atoi(line.c_str());
  line = line.substr(line.find(" ",line.find_first_not_of(" ")));
  z = atoi(line.c_str());
  printf("x = %i, y = %i, z = %i \n",x,y,z);

  //initialize
  n = x;
  m = y*8;
  //z is the number of boundary edges
   if (verbose) {
    printf("n %d, m %u, z %u\n", n, m/2,z);
    elt = timer();
  }

  srcs = new int[m];
  dsts = new int[m];
  boundary = new bool[n];

  //initialize boundary array of length n, set all false
  for(int i = 0; i < n; i++){
    boundary[i] = false;
  }

  //ignore the next x lines
  while(x --> 0){
    getline(infile,line);
  }
  getline(infile, line);

  //for the next y lines
  // read in the first 4 ints
  // create 8 edges from those ints, subtracting one from all values for 0-indexing
  unsigned int edge_index = 0;
  while( y --> 0){
    int node1 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" "));
    int node2 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" ", line.find_first_not_of(" ")));
    int node3 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" ", line.find_first_not_of(" ")));
    int node4 = atoi(line.c_str()) - 1;

    /*if(verbose){
      printf("%d %d\n",node1, node2);
      printf("%d %d\n",node2, node3);
      printf("%d %d\n",node3, node4);
      printf("%d %d\n",node4, node1);
    }*/

    srcs[edge_index] = node1;
    dsts[edge_index++] = node2;
    srcs[edge_index] = node2;
    dsts[edge_index++] = node1;
    srcs[edge_index] = node2;
    dsts[edge_index++] = node3;
    srcs[edge_index] = node3;
    dsts[edge_index++] = node2;
    srcs[edge_index] = node3;
    dsts[edge_index++] = node4;
    srcs[edge_index] = node4;
    dsts[edge_index++] = node3;
    srcs[edge_index] = node4;
    dsts[edge_index++] = node1;
    srcs[edge_index] = node1;
    dsts[edge_index++] = node4;

    getline(infile, line);
  }
  assert(edge_index == m);  

  
  //for the next z lines
  for(int i = 0; i < z; i++){
    int node1 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" "));
    int node2 = atoi(line.c_str()) - 1;
    // mark true all integers observed
    boundary[node1] = true;
    boundary[node2] = true;
    getline(infile, line);
  }

  infile.close();

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }

  return;

}


void count_components(graph* g, int* comps)
{
  int* counts = new int[g->n];
  for (int i = 0; i < g->n; ++i)
    counts[i] = 0;

  for (int v = 0; v < g->n; ++v)
    ++counts[comps[v]];

  int max_size = 0;
  int nontrivial = 0;
  int num_comps = 0;
  for (int i = 0; i < g->n; ++i) {
    if (counts[i] > 0) {
      ++num_comps;
      if (counts[i] > 1) 
        ++nontrivial;
      if (counts[i] > max_size)
        max_size = counts[i];
    }
  }

  int target_comp = -1;
  for (int i = 0; i < g->n; ++i){
    if (counts[i] > 0){
      if(counts[i] == max_size){
        target_comp = i;
      }
    }
  }

  int* srcs = new int[2*max_size];
  int* dsts = new int[2*max_size];
  int counter = 0;
  for (int i = 0; i < g->n; ++i){
    if (comps[i] == target_comp) {
      int* outs = out_vertices(g,i);
      int out_degree = out_degree(g,i);
      for(int j = 0; j<out_degree; j++){
        //printf("%d -- %d\n",i,outs[j]);
        // srcs[counter] = i;
        // dsts[counter++]=j;
        // srcs[counter] = j;
        // dsts[counter++] = i;
        counter += 2;
      }
    }
  }
  printf("Num edges in bcc: %d\n",counter);
  delete srcs;
  delete dsts;

  printf("Num comps: %d, max comp: %d, nontrivial: %d\n", 
    num_comps, max_size, nontrivial);
}

int find_max_comp(graph *g, int* comps, int*& counts){
  counts = new int[g->n];
  for (int i = 0; i < g->n; ++i)
    counts[i] = 0;

  for (int v = 0; v < g->n; ++v)
    ++counts[comps[v]];

  int max_size = 0;
  int nontrivial = 0;
  int num_comps = 0;
  for (int i = 0; i < g->n; ++i) {
    if (counts[i] > 0) {
      ++num_comps;
      if (counts[i] > 1) 
        ++nontrivial;
      if (counts[i] > max_size)
        max_size = counts[i];
    }
  }

  int target_comp = -1;
  for (int i = 0; i < g->n; ++i){
    if (counts[i] > 0){
      if(counts[i] == max_size){
        target_comp = i;
      }
    }
  }
  //delete [] counts

  return target_comp;
}

int* color_prop(graph* g)
{
  int* comp = new int[g->n];
  for (int i = 0;i < g->n; ++i)
    comp[i] = i;

  double elt = omp_get_wtime();

  int num_updates = 1;
  while (num_updates > 0) {
    num_updates = 0;

//#pragma omp parallel for schedule(guided) reduction(+:num_updates)
    for (int v = 0; v < g->n; ++v) {
      int* outs = out_vertices(g, v);
      int out_degree = out_degree(g, v);
      for (int j = 0; j < out_degree; ++j) {
        int u = outs[j];
        if (comp[u] > comp[v]) {
          comp[v] = comp[u];
          ++num_updates;
        }
      }
    }
  }

  elt = omp_get_wtime() - elt;
  printf("Time: %9.6lf\n", elt);

  //count_components(g, comp);

  //delete [] comp;
  return comp;
}





void create_csr(int n, unsigned m, 
  int* srcs, int* dsts,
  int*& out_array, unsigned*& out_degree_list,
  int& max_degree_vert, double& avg_out_degree)
{
  double elt = 0.0;
  if (verbose) {
    printf("Creating graph format\n");
    elt = timer();
  }

  out_array = new int[m];
  out_degree_list = new unsigned[n+1];
  unsigned* temp_counts = new unsigned[n];

#pragma omp parallel for
  for (unsigned i = 0; i < m; ++i)
    out_array[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    temp_counts[i] = 0;

  for (unsigned i = 0; i < m; ++i)
    ++temp_counts[srcs[i]];
  for (int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, n*sizeof(int));
  for (unsigned i = 0; i < m; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];
  delete [] temp_counts;

  unsigned max_degree = 0;
  max_degree_vert = -1;
  avg_out_degree = 0.0;
  for (int i = 0; i < n; ++i)
  {
    unsigned degree = out_degree_list[i+1] - out_degree_list[i];
    avg_out_degree += (double)degree;
    if (degree > max_degree)
    {
      max_degree = degree;
      max_degree_vert = i;
    }
  }
  avg_out_degree /= (double)n;
  assert(max_degree_vert >= 0);
  assert(avg_out_degree >= 0.0);

  
  if (verbose) {
    printf("max degree vert: %d, max degree: %d, avg degree %lf\n", max_degree_vert, 
      (out_degree_list[max_degree_vert+1] - out_degree_list[max_degree_vert]),
      avg_out_degree);
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }
}

void remove_ice_islands(graph& g, bool* boundaries){
  //use color propagation to find the separate components of the graph
  int* comps = color_prop(&g);
  //find the biggest component
  int* counts = NULL;
  int max_comp = find_max_comp(&g,comps,counts);

  //make new labels so that the nodes in the max component are
  // 0-indexed and sequential. Also need to find the number of edges
  int n = counts[max_comp];
  int* new_labels = new int[g.n];
  unsigned int m = 0;
  int label_counter = 0;
  graph* gp = &g;
  for (int i = 0; i < g.n; ++i){
    if (comps[i] == max_comp) {
      new_labels[i] = label_counter++; 
      int* outs = out_vertices(gp,i);
      int out_degree = out_degree(gp,i);
      for(int j = 0; j<out_degree; j++){
        if(comps[outs[j]] == max_comp)
          m += 1;
      }
    } else {
      new_labels[i] = -1;
    }
  }
  assert(label_counter == n);

  int* srcs = new int[m];
  int* dsts = new int[m];
  int edge_counter = 0;
  for(int i = 0; i < g.n; i++){
    //if the node has a -1 for a new label, it isn't in the max component
    if (new_labels[i] >-1){ //AND it's a boundary (after I ensure this works)
      int * outs = out_vertices(gp,i);
      int out_degree = out_degree(gp,i);
      //look at all the neighbors of this node
      for(int j=0; j< out_degree; j++){
        if(new_labels[outs[j]] >-1){
          //printf("%d %d \n",new_labels[i],new_labels[outs[j]]);
          srcs[edge_counter] = new_labels[i];
          dsts[edge_counter++] = new_labels[outs[j]];
        }
      }
    }
  }
  assert(edge_counter == m);

  // int* out_array;
  // unsigned* out_degree_list;
  int max_degree_vert;
  double avg_out_degree;

   create_csr(n, m, srcs, dsts, 
    g.out_array, g.out_degree_list, max_degree_vert, avg_out_degree);

   g = {n, m, g.out_array, g.out_degree_list, max_degree_vert, avg_out_degree};
}

void output_bcc(graph* g, int* bcc_maps, char* output_file)
{
  double elt = 0.0;
  if (verbose) {
    printf("Outputting mapping as %s\n", output_file);
    elt = timer();
  }

  std::ofstream outfile;
  outfile.open(output_file);

  for (unsigned i = 0; i < g->m; ++i)
    outfile << bcc_maps[i] << std::endl;

  outfile.close();  

  if (verbose) {
    elt = timer() - elt;
    printf("\tDone: %9.6lf\n", elt);
  }
}




int main(int argc, char* argv[])
{
  srand(time(0));
  setbuf(stdout, NULL);  

  if (argc < 3)
    print_usage(argv);

  bool run_bfs = false;
  bool run_color = false;

  verbose = false;
  debug = false;
  random_start = false;
  
  bool output = false; 
  char output_file[1024];
  char graph_file[1024];
  strcpy(graph_file, argv[1]);
  int max_levels = MAX_LEVELS;

  char c;
  while ((c = getopt (argc, argv, "hbcm:vdro:")) != -1) {
    switch (c) {
      case 'h': print_usage_full(argv); break;
      case 'b': run_bfs = true; break;
      case 'c': run_color = true;  break;
      case 'm': max_levels = atoi(optarg); break;
      case 'v': verbose = true; break;
      case 'd': debug = true; break;
      case 'r': random_start = true; break;
      case 'o': 
        output = true;
        strcpy(output_file, optarg);
        break;
      default:
        fprintf(stderr, "Input argument format error");
        exit(1);
    }
  }

  printf("Starting BCC decomposition...\n");

  int* srcs;
  int* dsts;
  bool* boundaries;
  int n;
  unsigned m;
  int z;
  int* out_array;
  unsigned* out_degree_list;
  int max_degree_vert;
  double avg_out_degree;

  //read_edge(graph_file, n, m, srcs, dsts);
  read_edge_mesh(graph_file, n, m, z, srcs, dsts, boundaries);
  create_csr(n, m, srcs, dsts, 
    out_array, out_degree_list, max_degree_vert, avg_out_degree);
  graph g = {n, m, out_array, out_degree_list, max_degree_vert, avg_out_degree};
  //color_prop(&g);
  remove_ice_islands(g,boundaries);
  assert(g.n < n);

  delete [] srcs;
  delete [] dsts;

  int* bcc_maps = new int[g.m];

  if (run_bfs)
    bcc_bfs(&g, bcc_maps, max_levels);
  if (run_color)
    bcc_color(&g, bcc_maps);

  if (output)
    output_bcc(&g, bcc_maps, output_file);

  delete [] bcc_maps;
  delete [] out_degree_list;
  delete [] out_array;

  return 0;
}

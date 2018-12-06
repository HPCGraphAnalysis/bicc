#include <mpi.h>
#include <omp.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <queue>

#include "dist_graph.h"
#include "bicc_dist.h"
#include "label_prop.h"

void read_edge_mesh(char* filename, int &n, unsigned &m, int*& srcs, int*& dsts, int*& grounded_flags, int ground_sensitivity){
  std::ifstream infile;
  std::string line;
  infile.open(filename);
  
  //ignore first line
  std::getline(infile, line);
  
  std::getline(infile, line);
  int x = atoi(line.c_str());
  line = line.substr(line.find(" "));
  int y = atoi(line.c_str());
  line = line.substr(line.find(" ", line.find_first_not_of(" ")));
  int z = atoi(line.c_str());
  
  //initialize
  n = x;
  m = y*8;
  //z is the number of floating boundary edges
  
  srcs = new int[m];
  dsts = new int[m];
  //ignore the next x lines
  while(x-- > 0){
    std::getline(infile,line);
  }
  std::getline(infile,line);
  
  //create the final_ground_flags array, initially everything is floating
  int* final_ground_flags = new int[n];
  for(int i = 0; i < n; i++){
    final_ground_flags[i] = 0;
  }
  int edge_index = 0;
  //for the next y lines
  //read in the first 4 ints
  //create 8 edges from those ints, subtractinv one from all values for 0-indexing
  while(y-- > 0){
    int node1 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" "));
    int node2 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" ", line.find_first_not_of(" ")));
    int node3 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" ", line.find_first_not_of(" ")));
    int node4 = atoi(line.c_str()) - 1;

    // set the final grounding
    int grounding = grounded_flags[node1] + grounded_flags[node2] + grounded_flags[node3] + grounded_flags[node4];
    if(grounding >= ground_sensitivity){
      final_ground_flags[node1] += grounded_flags[node1];
      final_ground_flags[node2] += grounded_flags[node2];
      final_ground_flags[node3] += grounded_flags[node3];
      final_ground_flags[node4] += grounded_flags[node4];
    }

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

    std::getline(infile, line);
  }
  //assert(edge_index == m);
  
  infile.close();
  
  //delete old grounding flags, and swap them for the new ones
  if(ground_sensitivity > 1){
    delete [] grounded_flags;
    grounded_flags = final_ground_flags;
  } else {
    delete [] final_ground_flags;
  }
  return;
}

void read_boundary_file(char* filename, int n, int*& boundary_flags){
  std::ifstream fin(filename);
  if(!fin){
    std::cout<<"Unable to open file "<<filename<<"\n";
    exit(0);
  }
  std::string throwaway;
  fin>>throwaway>>throwaway;
  int nodes, skip2, arrlength;
  fin>>nodes>>skip2>>arrlength;
  for(int i = 0; i <= nodes; i++){
    std::getline(fin,throwaway);
  }
  for(int i = 0; i < skip2; i++){
    std::getline(fin,throwaway);
  }
  boundary_flags = new int[n];
  for(int i = 0; i < n; i++){
    boundary_flags[i] = 0;
  }
  int a, b;
  //nodes that we see more than twice are potential articulation points
  while(fin>>a>>b>>throwaway){
    boundary_flags[a-1] += 1;
    boundary_flags[b-1] += 1;
  }
  for(int i = 0; i < n; i++){
    boundary_flags[i] = boundary_flags[i] > 2;
  }
}

void read_grounded_file(char* filename, int&n, int*& grounded_flags){
  std::ifstream fin(filename);
  if(!fin){
    std::cout<<"Unable to open "<<filename<<"\n";
    exit(0);
  }
  //the first number is the number of vertices
  fin>>n;
  grounded_flags = new int[n];
  //the rest of the numbers are basal friction data
  for(int i = 0; i < n; i++){
    float gnd;
    fin>>gnd;
    grounded_flags[i] = (gnd > 0.0);
  }
}

void create_csr(int n, unsigned m, int* srcs, int* dsts, int*& out_array, unsigned*& out_degree_list, int& max_degree_vert, double& avg_out_degree){
  out_array = new int[m];
  out_degree_list = new unsigned[n+1];
  unsigned* temp_counts = new unsigned[n];
  
  for(unsigned i = 0; i < m; ++i)
    out_array[i] = 0;
  for(int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;
  for(int i = 0; i < n; ++i)
    temp_counts[i] = 0;
 
  for(unsigned i = 0; i < m; ++i)
    ++temp_counts[srcs[i]];
  for(int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, n*sizeof(int));
  for(unsigned i = 0; i < m; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];
  delete [] temp_counts;
  
  unsigned max_degree = 0;
  max_degree_vert = -1;
  avg_out_degree = 0.0;
  for(int i = 0; i < n; ++i){
    unsigned degree = out_degree_list[i+1] - out_degree_list[i];
    avg_out_degree += (double) degree;
    if(degree > max_degree){
      max_degree = degree;
      max_degree_vert = i;
    }
  }
  avg_out_degree /= (double) n;
}

extern int procid, nprocs;

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if(argc < 5){
    if(procid == 0){
      std::cout<<"Usage: mpirun -np <num_procs> ice <mesh_file> <boundary_file> <grounded_file> <ground_sensitivity>\n";
      exit(0);
    }
  }
  
  // create necessary variables for the global graph
  int n;
  int* grounded_flags;
  int *srcs, *dsts;
  unsigned m;
  int* boundary_flags;
  
  //start timing for file reading here
  if(procid == 0){
    read_grounded_file(argv[3],n,grounded_flags);
    
    int ground_sensitivity = atoi(argv[4]);
    read_edge_mesh(argv[1],n,m,srcs,dsts,grounded_flags,ground_sensitivity);
    
    read_boundary_file(argv[2],n,boundary_flags);
    
  }
  
  //stop file read time
  
  //start distribution timer
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  //allocate memory to the arrays
  if(procid != 0){
    grounded_flags = new int[n];
    srcs = new int[m];
    dsts = new int[m];
    boundary_flags = new int[n];
  }
  
  //broadcast the global data
  MPI_Bcast(grounded_flags,n,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(boundary_flags,n,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(srcs,m,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(dsts,m,MPI_INT,0,MPI_COMM_WORLD);
  
  //select locally owned vertices
  int n_local = 0;
  int np = nprocs;
  //simple block partitioning
  n_local = n/np + (procid < (n%np));
  int local_offset = std::min(procid,n%np)*(n/np + 1) + std::max(0, procid - (n%np))*(n/np);
  
  std::cout<<procid<<": vertices go from "<<local_offset<<" to "<<local_offset+n_local-1<<"\n";
   
  int *copies = new int[n];
  int *localOwned = new int[n];
  int *newId = new int[n];
  int new_id_counter = 0; 
  
  for(int i = 0; i < n; i++){
    copies[i] = 0;
    if(i >= local_offset && i < local_offset + n_local){
      localOwned[i] = 1;
      newId[i] = new_id_counter++;
    } else {
      localOwned[i] = 0;
      newId[i] = -1;
    }
  }
  
  int *localSrcs = new int[m];
  int *localDsts = new int[m];
  unsigned int localEdgeCounter = 0;
  int numcopies = 0;
  
  for(unsigned i = 0; i < m; i++){
    if(localOwned[srcs[i]]){
      localSrcs[localEdgeCounter] = newId[srcs[i]];
      if(!localOwned[dsts[i]]){
        if(copies[dsts[i]] == 0){
          copies[dsts[i]] = 1;
          numcopies++;
        }
        if(newId[dsts[i]] < 0) newId[dsts[i]] = new_id_counter++;
      }
      localDsts[localEdgeCounter++] = newId[dsts[i]];
    } else if(localOwned[dsts[i]]){
      localDsts[localEdgeCounter] = newId[dsts[i]];
      if(!localOwned[srcs[i]]){
        if(copies[srcs[i]] == 0){
          copies[srcs[i]] = 1;
          numcopies++;
        }
        if(newId[srcs[i]] < 0) newId[srcs[i]] = new_id_counter++;
      }
      localSrcs[localEdgeCounter++] = newId[srcs[i]];
    }
  }
  
  //make new grounding/boundary arrays for just the local vertices (owned+copies)
  uint64_t* localGrounding = new uint64_t[n_local + numcopies];
  uint64_t* localBoundaries = new uint64_t[n_local + numcopies];
  for(int i = 0; i < n_local+numcopies; i++){
    localGrounding[i] = 0;
    localBoundaries[i] = 0;
  }
  for(int i = 0; i < n; i++){
    if(newId[i] > -1){
      localGrounding[newId[i]] = grounded_flags[i];
      localBoundaries[newId[i]] = boundary_flags[i];
    }
  }
  
  //create gids for unmapping
  uint64_t* gids = new uint64_t[n_local + numcopies];
  for(int i = 0; i < n; i++){
    if(newId[i] > -1){
      gids[newId[i]] = i;
      std::cout<<"Task "<<procid<<" local "<<newId[i]<<" is global "<<i<<"\n";
    }
  }
 
  
   
  int* out_array;
  unsigned* out_degree_list;
  int max_degree_vert;
  double avg_out_degree;
  create_csr(new_id_counter,localEdgeCounter, localSrcs,localDsts,out_array, out_degree_list, max_degree_vert, avg_out_degree);
   
  for(int i = 0; i < n_local; i ++){
    for(int j = out_degree_list[i]; j < out_degree_list[i+1]; j++){
      std::cout<<"Task "<<procid<<": "<< i <<" - "<<out_array[j]<<"\n";
    }
  }
   
  dist_graph_t* g = new dist_graph_t;
  
  uint64_t* local_offsets = new uint64_t[n_local+1];
  uint64_t* local_adjs = new uint64_t[localEdgeCounter];
  
  for(int i = 0; i < n_local+1; i++){
    local_offsets[i] = out_degree_list[i];
  }
  
  for(int i = 0; i < localEdgeCounter; i++){
    local_adjs[i] = out_array[i];
  }
  
  create_graph(g, n, m, n_local, localEdgeCounter, local_offsets, local_adjs, gids);
  std::cout<<"Task "<<procid<<" BEFORE RELABELING\n"; 
  for(int i = 0; i < g->n_local; i++){
    int out_degree = out_degree(g,i);
    uint64_t* outs = out_vertices(g,i);
    for(int j = 0; j < out_degree; j++){
      int neighbor = g->local_unmap[outs[j]];
      //if(outs[j] >= g->n_local) neighbor = g->ghost_unmap[outs[j]-g->n_local];
      std::cout<<"Task "<<procid<<": "<<g->local_unmap[i]<<" - "<<neighbor<<"\n";
    }
  }
  relabel_edges(g);
  std::cout<<"AFTER RELABELING\n";
  for(int i = 0; i < g->n_local; i++){
    int out_degree = out_degree(g,i);
    uint64_t* outs = out_vertices(g,i);
    for(int j = 0; j < out_degree; j++){
      int neighbor = g->local_unmap[outs[j]];
      if(outs[j] >= g->n_local) neighbor = g->ghost_unmap[outs[j]-g->n_local];
      std::cout<<"Task "<<procid<<": "<<g->local_unmap[i]<<" - "<<neighbor<<"\n";
    }
  }
  
  std::cout<<"Task "<<procid<<": total vertices = "<<g->n_total<<"\n";
  int** labels = new int*[g->n_total];
  for(int i = 0; i < g->n_total; i++){
    labels[i] = new int[5];
    labels[i][0] = -1;
    labels[i][1] = -1;
    labels[i][2] = -1;
    labels[i][3] = -1;
    labels[i][4] = -1;
  }
  std::queue<int> reg_frontier;
  std::queue<int> art_frontier;
  //set labels for grounded nodes.
  for(int i = 0; i < n_local; i++){
    if(localGrounding[i]){
      std::cout<<"Task "<<procid<<": grounding vtx "<<i<<"\n";
      labels[i][0] = g->local_unmap[i];
      labels[i][1] = g->local_unmap[i];
      if(localBoundaries[i]){
        art_frontier.push(i);
      }else{ 
        reg_frontier.push(i);
      }
    }
  }
  
  int* removed = propagate(g, reg_frontier, art_frontier, labels, localBoundaries);

  for(int i = 0; i < g->n_local; i++){
    if(removed[i] > -2){
      std::cout<<procid<<": removed "<<g->local_unmap[i]<<"\n";
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

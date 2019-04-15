#include <mpi.h>
#include <omp.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <queue>
#include <cstring>

#include "dist_graph.h"
#include "bicc_dist.h"
#include "label_prop.h"
#include "comms.h"
#include "io_pp.h"

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

void create_dist_graph_from_binadjfile(char* filename, dist_graph_t* g){
  MPI_File file;
  MPI_Status status;
  int rc = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,MPI_INFO_NULL, &file);

  if(rc){
    std::cout<<"Rank"<<procid<<": Failed to open "<<filename<<"\n";
  }

  uint64_t global_verts = 0;
  MPI_File_read(file, &global_verts, 1, MPI_UNSIGNED_LONG_LONG, &status);
  //n_local = n/np + (procid < (n%np));
  //int local_offset = std::min(procid,n%np)*(n/np + 1) + std::max(0, procid - (n%np))*(n/np);
  //int* proc_offsets = new int[np];
  //for(int i = 0; i < np; i++){
  //  proc_offsets[i] = std::min(i,n%np)*(n/np + 1) + std::max(0, i - (n%np))*(n/np);
  //}
  
  uint64_t n_local = global_verts/(uint64_t)nprocs + ((uint64_t)procid < (global_verts%(uint64_t)nprocs));
  uint64_t local_offset = std::min((uint64_t)procid, global_verts%(uint64_t)nprocs)*(global_verts/(uint64_t)nprocs + 1);
  local_offset += ((uint64_t)procid > (global_verts%(uint64_t)nprocs))?((uint64_t)procid-(global_verts%(uint64_t)nprocs))*(global_verts/(uint64_t)nprocs):0;
  
  uint64_t *proc_offsets = new uint64_t[nprocs];
  for(int i = 0; i < nprocs; i++){
    proc_offsets[i] = std::min((uint64_t)i, global_verts%nprocs)*(global_verts/nprocs + 1);
    proc_offsets[i] += (i > (global_verts%nprocs))?(i - (global_verts%nprocs))*(global_verts/nprocs):0;
  }

  std::cout<<"Rank"<<procid<<": vertices start at "<<local_offset<<" and go to "<<local_offset + n_local - 1<<"\n";

  //keep reading ints until we count enough zeros
  uint64_t tmp = 4;
  MPI_File_read(file,&tmp, 1, MPI_UNSIGNED_LONG_LONG, &status);
  //the number after the global vertex count is a sentinel zero
  if(tmp != 0ull) std::cout<<"Rank"<<procid<<": did not read 0 after global count\n";

  //store the file offset, to mark where to restart file reading after arrays are allocated
  uint64_t start_offset = 2ull;
  uint64_t zero_count = 0ull;
  uint64_t nlocal_edges = 0ull;
  uint64_t nghost_guess = 0ull;
  //find the start point for this processor's vertices
  while(zero_count < local_offset){
    MPI_File_read(file,&tmp, 1, MPI_UNSIGNED_LONG_LONG, &status);
    start_offset++;
    if(tmp == 0ull) zero_count++;
  }
  
  //find the end_offset
  uint64_t end_offset = start_offset;
  while(zero_count + 1 >= local_offset && zero_count < local_offset + n_local){
    MPI_File_read(file, &tmp, 1, MPI_UNSIGNED_LONG_LONG, &status);
    end_offset++;
    if(tmp != 0ull) nlocal_edges++;
    if(tmp == 0ull) zero_count++;
    else if(tmp < local_offset || tmp >= local_offset + n_local - 1) nghost_guess++;
  }

  //initialize the global to local map for ease of determining ghosts
  fast_map* global_to_local = (struct fast_map*)malloc(sizeof(struct fast_map));
  init_map(global_to_local, (n_local+nlocal_edges)*2);
  
  //allocate local arrays
  uint64_t* offsets = new uint64_t[n_local+1];
  uint64_t* adjacencies = new uint64_t[nlocal_edges];
  uint64_t* gids = new uint64_t[n_local + nghost_guess];

  std::cout<<"Rank"<<procid<<": start_offset = "<<start_offset<<" end_offset = "<<end_offset<<"\n";
  uint64_t new_id_counter = n_local;

  //set local gids
  for(int i = 0; i < n_local; i++){
    set_value(global_to_local, local_offset+i,i);
    gids[i] = local_offset+i;
  }

  //read in the adjacencies, build required arrays
  MPI_File_seek(file, start_offset*8, MPI_SEEK_SET);
  uint64_t curr_vtx = 0;
  uint64_t curr_degree = 0;
  uint64_t nghosts = 0;
  offsets[curr_vtx++] = curr_degree;
  for(uint64_t i = start_offset; i < end_offset; i++){
    MPI_File_read(file, &tmp, 1, MPI_UNSIGNED_LONG_LONG, &status);
    if(tmp == 0){
      //we're done with the current vertex entry
      offsets[curr_vtx++] = curr_degree;
    } else {
      //see if the current vertex has been recorded in the gid structures
      uint64_t lid = get_value(global_to_local,tmp-1);
      if(lid == NULL_KEY) {
        nghosts++;
	set_value(global_to_local, tmp-1, new_id_counter);
	adjacencies[curr_degree++] = new_id_counter;
	gids[new_id_counter++] = tmp-1;
      }else {
        adjacencies[curr_degree++] = lid;
      }
    }
  }
  uint64_t* ghost_unmap = gids+n_local;
  //do the ghost_unmap stuff
  uint64_t* ghost_tasks = new uint64_t[nghosts];
  for(int i = 0; i < nghosts; i++){
    bool tasked = false;
    for(int j = 0; j < nprocs; j++){
      if(ghost_unmap[i] < proc_offsets[j] && !tasked){
        ghost_tasks[i] = j-1;
	tasked = true;
	break;
      }
    }
    if(tasked == false) ghost_tasks[i] = nprocs-1;
  }

  g->n = global_verts;
  g->m = nlocal_edges;
  g->m_local = nlocal_edges;
  g->n_local = n_local;
  g->n_offset = local_offset;
  g->n_ghost = nghosts;
  g->n_total = nghosts+n_local;
  g->out_edges = adjacencies;
  g->out_degree_list = offsets;
  g->local_unmap = gids;
  g->ghost_unmap = ghost_unmap;
  g->ghost_tasks = ghost_tasks;
  g->map = global_to_local;

  MPI_File_close(&file);
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
  double elt = 0.0;
  elt = timer();

  if(procid == 0){
    read_grounded_file(argv[3],n,grounded_flags);
    
    //int ground_sensitivity = atoi(argv[4]);
    //read_edge_mesh(argv[1],n,m,srcs,dsts,grounded_flags,ground_sensitivity);
    
    read_boundary_file(argv[2],n,boundary_flags);
    int artpts = 0;
    for(int i = 0; i < n; i++){
      artpts+=boundary_flags[i];
    } 
    std::cout<<"found "<<artpts<<" potential artpts\n";
  }
  
  //stop file read time
  double fileread_time = timer() - elt;
  double fileread_time_max = 0.0;
  double fileread_time_min = 0.0;
  double fileread_time_mean = 0.0;
  //reduce times across processors
  MPI_Reduce(&fileread_time,&fileread_time_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&fileread_time,&fileread_time_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&fileread_time,&fileread_time_mean,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if(procid == 0){
    fileread_time_mean /= nprocs;
    printf("File Read Timing:\n");
    printf("\tMin:\t%f \n",fileread_time_min);
    printf("\tMax:\t%f \n",fileread_time_max);
    printf("\tMean: \t%f \n",fileread_time_mean);
  }

  
  //start distribution timer
  elt = timer();
  
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  //allocate memory to the arrays
  if(procid != 0){
    grounded_flags = new int[n];
    //srcs = new int[m];
    //dsts = new int[m];
    boundary_flags = new int[n];
  }
  
  //broadcast the global data
  MPI_Bcast(grounded_flags,n,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(boundary_flags,n,MPI_INT,0,MPI_COMM_WORLD);
  //MPI_Bcast(srcs,m,MPI_INT,0,MPI_COMM_WORLD);
  //MPI_Bcast(dsts,m,MPI_INT,0,MPI_COMM_WORLD);
  
  //select locally owned vertices
  //int n_local = 0;
  //int np = nprocs;
  //simple block partitioning
  //n_local = n/np + (procid < (n%np));
  //int local_offset = std::min(procid,n%np)*(n/np + 1) + std::max(0, procid - (n%np))*(n/np);
  //int* proc_offsets = new int[np];
  //for(int i = 0; i < np; i++){
  //  proc_offsets[i] = std::min(i,n%np)*(n/np + 1) + std::max(0, i - (n%np))*(n/np);
  //}
  
  //std::cout<<procid<<": vertices go from "<<local_offset<<" to "<<local_offset+n_local-1<<"\n";
   
  //int *copies = new int[n];
  //int *localOwned = new int[n];
  //int *newId = new int[n];
  //int new_id_counter = 0; 
  
  //for(int i = 0; i < n; i++){
  //  copies[i] = 0;
  //  if(i >= local_offset && i < local_offset + n_local){
  //    localOwned[i] = 1;
  //    newId[i] = new_id_counter++;
  //  } else {
  //    localOwned[i] = 0;
  //    newId[i] = -1;
  //  }
  //}
  
  //int *localSrcs = new int[m];
  //int *localDsts = new int[m];
  //unsigned int localEdgeCounter = 0;
  //int numcopies = 0;
  
  //for(unsigned i = 0; i < m; i++){
  //  if(localOwned[srcs[i]]){
  //    localSrcs[localEdgeCounter] = newId[srcs[i]];
  //    if(!localOwned[dsts[i]]){
  //      if(copies[dsts[i]] == 0){
  //        copies[dsts[i]] = 1;
  //        numcopies++;
  //      }
  //      if(newId[dsts[i]] < 0) newId[dsts[i]] = new_id_counter++;
  //    }
  //    localDsts[localEdgeCounter++] = newId[dsts[i]];
  //  } else if(localOwned[dsts[i]]){
  //    localDsts[localEdgeCounter] = newId[dsts[i]];
  //    if(!localOwned[srcs[i]]){
  //      if(copies[srcs[i]] == 0){
  //        copies[srcs[i]] = 1;
  //        numcopies++;
  //      }
  //      if(newId[srcs[i]] < 0) newId[srcs[i]] = new_id_counter++;
  //    }
  //    localSrcs[localEdgeCounter++] = newId[srcs[i]];
  //  }
  //}
  
  //make new grounding/boundary arrays for just the local vertices (owned+copies)
  //uint64_t* localGrounding = new uint64_t[n_local + numcopies];
  //uint64_t* localBoundaries = new uint64_t[n_local + numcopies];
  //for(int i = 0; i < n_local+numcopies; i++){
  //  localGrounding[i] = 0;
  //  localBoundaries[i] = 0;
  //}
  //for(int i = 0; i < n; i++){
  //  if(newId[i] > -1){
  //    localGrounding[newId[i]] = grounded_flags[i];
  //    localBoundaries[newId[i]] = boundary_flags[i];
  //  }
  //}
   
  //create gids for unmapping
  //uint64_t* gids = new uint64_t[n_local + numcopies];
  //for(int i = 0; i < n; i++){
  //  if(newId[i] > -1){
  //    gids[newId[i]] = i;
      //std::cout<<"Task "<<procid<<" local "<<newId[i]<<" is global "<<i<<"\n";
  //  }
  //}
 
  
   
  //int* out_array;
  //unsigned* out_degree_list;
  //int max_degree_vert;
  //double avg_out_degree;
  //create_csr(new_id_counter,localEdgeCounter, localSrcs,localDsts,out_array, out_degree_list, max_degree_vert, avg_out_degree);
  
  //for(int i = 0; i < n_local; i ++){
  //  for(int j = out_degree_list[i]; j < out_degree_list[i+1]; j++){
      //std::cout<<"Task "<<procid<<": "<< gids[i] <<" - "<<gids[out_array[j]]<<"\n";
  //  }
  //}
  //std::cout<<"Task "<<procid<<"'s local graph\n";
  //for(int i =0; i < localEdgeCounter; i++){
    //std::cout<<"Task "<<procid<<": "<<localSrcs[i]<<" - "<<localDsts[i]<<"\n";
  //}
  //uint64_t* local_offsets = new uint64_t[n_local+1];
  //uint64_t* local_adjs = new uint64_t[localEdgeCounter];
  //uint64_t* local_unmap = new uint64_t[n_local];
  //uint64_t* ghost_unmap = new uint64_t[numcopies];
  //uint64_t* ghost_tasks = new uint64_t[numcopies];
  
  //for(int i = 0; i < n_local+1; i++){
  //  local_offsets[i] = out_degree_list[i];
  //}
  
  //for(int i = 0; i < localEdgeCounter; i++){
  //  local_adjs[i] = out_array[i];
  //}
  
  //for(int i = 0; i < n_local; i++){
  //  local_unmap[i] = gids[i];
  //}  
  
  //for(int i = 0; i < numcopies; i++){
  //  ghost_unmap[i] = gids[n_local+i];
  //  bool tasked = false;
  //  for(int j = 0; j < np; j++){
  //    if(ghost_unmap[i] < proc_offsets[j] && !tasked){
        //std::cout<<"Task "<<procid<<": vertex "<<ghost_unmap[i]<<" belongs to proc "<<j-1<<"\n";
  //      ghost_tasks[i] = j-1;
  //      tasked = true;
  //    }
  //  }
  //  if(tasked == false){
  //    ghost_tasks[i] = np-1;
      //std::cout<<"Task "<<procid<<": vertex "<<ghost_unmap[i]<<" belongs to proc "<<np-1<<"\n";
  //  }
  //}
  
  
  
  dist_graph_t g;
  /*g.n = n;
  g.m = m;
  g.m_local = localEdgeCounter;
  g.n_local = n_local;
  g.n_offset = local_offset;
  g.n_ghost = numcopies;
  g.n_total = n_local + numcopies;
  g.out_edges = local_adjs;
  g.out_degree_list = local_offsets;
  g.local_unmap = local_unmap;
  g.ghost_unmap = ghost_unmap;
  g.ghost_tasks = ghost_tasks;
  g.map = (struct fast_map*)malloc(sizeof(struct fast_map));
  init_map(g.map, (n_local+localEdgeCounter)*2);
  for(uint64_t i = 0; i < n_local + numcopies; i++){
    uint64_t vert = gids[i];
    set_value(g.map, vert, i);
    //get_value(g.map,vert);
  }*/
    
  dist_graph_t* gp = &g;
  
  create_dist_graph_from_binadjfile(argv[1],gp);
    
  
  uint64_t* localGrounding = new uint64_t[gp->n_total];
  uint64_t* localBoundaries = new uint64_t[gp->n_total];
  for(int i = 0; i < gp->n_total; i++){
    localGrounding[i] = 0;
    localBoundaries[i] = 0;
  }
  for(int i = 0; i < n; i++){
    uint64_t vert = get_value(gp->map,i);
    if( vert != NULL_KEY ){
      localGrounding[vert] = grounded_flags[i];
      localBoundaries[vert] = boundary_flags[i];
    }
  }
  delete [] grounded_flags;
  delete [] boundary_flags;

  for(int i = 0; i < gp->n_local; i++){
    int out_degree = out_degree(gp,i);
    uint64_t* outs = out_vertices(gp,i);
    for(int j = 0; j < out_degree; j++){
      int neighbor = gp->local_unmap[outs[j]];
      if(outs[j] >= gp->n_local) neighbor = gp->ghost_unmap[outs[j]-gp->n_local];
      //std::cout<<"Task "<<procid<<": "<<gp->local_unmap[i]<<" - "<<neighbor<<"\n";
    }
  }
  //stop distribution timer
  double distribution_time = timer()-elt;
  double distribution_time_min = 0.0;
  double distribution_time_max = 0.0;
  double distribution_time_mean = 0.0;
  //reduce times across processors
  MPI_Reduce(&distribution_time,&distribution_time_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&distribution_time,&distribution_time_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&distribution_time,&distribution_time_mean,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if(procid == 0){
    distribution_time_mean /= nprocs;
    printf("Distribution Timing:\n");
    printf("\tMin:\t%f \n",distribution_time_min);
    printf("\tMax:\t%f \n",distribution_time_max);
    printf("\tMean: \t%f \n",distribution_time_mean);
  }
  

  //start solving timer
  elt = timer();
  //std::cout<<"Task "<<procid<<": total vertices = "<<g.n_total<<"\n";
  int** labels = new int*[g.n_total];
  for(int i = 0; i < g.n_total; i++){
    labels[i] = new int[5];
    labels[i][0] = -1;
    labels[i][1] = -1;
    labels[i][2] = -1;
    labels[i][3] = -1;
    labels[i][4] = -1;
  }
  //std::cout<<"Task "<<procid<<": done creating labels\n";
  std::queue<int> reg_frontier;
  std::queue<int> art_frontier;
  //set labels for grounded nodes.
  for(int i = 0; i < g.n_local; i++){
    if(localGrounding[i]){
      //std::cout<<"Task "<<procid<<": grounding vtx "<<i<<"\n";
      labels[i][0] = g.local_unmap[i];
      labels[i][1] = g.local_unmap[i];
      if(localBoundaries[i]){
        art_frontier.push(i);
      }else{ 
        reg_frontier.push(i);
      }
    }
  }
  //std::cout<<"Task "<<procid<<": calling propagate\n";
  int* removed = propagate(gp, reg_frontier, art_frontier, labels, localBoundaries);
  //std::cout<<"Task "<<procid<<": finishing propagate\n";
  for(int i = 0; i < gp->n_local; i++){
    if(removed[i] > -2){
      std::cout<<procid<<": removed "<<gp->local_unmap[i]+1<<"\n";
    }
  }
  //stop and report solving times
  //reduce times across processors
  double solve_time = timer() - elt;
  double solve_time_min = 0.0;
  double solve_time_max = 0.0;
  double solve_time_mean = 0.0;

  MPI_Reduce(&solve_time,&solve_time_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&solve_time,&solve_time_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&solve_time,&solve_time_mean,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if(procid == 0){
    solve_time_mean /= nprocs;
    printf("Solve Timing:\n");
    printf("\tMin:\t%f \n",solve_time_min);
    printf("\tMax:\t%f \n",solve_time_max);
    printf("\tMean: \t%f \n",solve_time_mean);
  }
  

  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

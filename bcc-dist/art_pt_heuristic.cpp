/* insert copyright stuff here*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <queue>

#include "dist_graph.h"
//#include "art_pt_heuristic_comms.h" <-This may be added in the future
#include "art_pt_heuristic.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;

void init_queue_nontree(dist_graph_t* g, std::queue<int> &q, uint64_t* parents,uint64_t* levels, int* visited_edges){
  for(int i = 0; i < g->n_local; i++){
    //go through the local nodes, and look through all connections that are not parent-child
    int out_degree = out_degree(g,i);
    uint64_t* outs = out_vertices(g,i);
    for(int j = 0; j < out_degree; j++){
      int neighbor = outs[j];
      int global_neighbor = neighbor;
      if(neighbor >= g->n_local) global_neighbor = g->ghost_unmap[global_neighbor-g->n_local];
      else global_neighbor = g->local_unmap[global_neighbor];
      int global_current = g->local_unmap[i];
      //printf("Checking edge between %d (parent: %d) and %d (parent: %d)\n",global_current,parents[i],global_neighbor,parents[neighbor]);
      if(parents[neighbor] != global_current && parents[i] != global_neighbor){
        //to prevent multiple traversals for the same edge, only put the entry on if the current node is smaller than its neighbor.
        if(global_current < global_neighbor) {
          //printf("nontree edge found from %d to %d\n",global_current,global_neighbor);
          //mark this edge as visited
          for(int k = g->out_degree_list[i]; k < g->out_degree_list[i+1]; k++){
            if(g->out_edges[k] == neighbor){
              printf("Task %d: marking edge from %d to %d as visited\n",procid,i,g->out_edges[k]);
              visited_edges[k] = 1;
            }
          }
          for(int k = g->out_degree_list[neighbor]; k < g->out_degree_list[neighbor+1]; k++){
            if(g->out_edges[k] == i){
              printf("Task %d: marking edge from %d to %d as visited\n",procid,neighbor,g->out_edges[k]);
              visited_edges[k] = 1;
            }
          }
          //mark i to parent[i](both ways)
          int local_parent_i = get_value(g->map, parents[i]);
          for(int k = g->out_degree_list[i]; k < g->out_degree_list[i+1]; k++){
            if(g->out_edges[k] == local_parent_i){
              printf("Task %d: marking edge from %d to %d as visited\n",procid,i,g->out_edges[k]);
              visited_edges[k] = 1;
            }
          }
          for(int k = g->out_degree_list[local_parent_i]; k < g->out_degree_list[local_parent_i+1]; k++){
            if(g->out_edges[k] == i){
              printf("Task %d: marking edge from %d to %d as visited\n",procid,local_parent_i,g->out_edges[k]);
              visited_edges[k] = 1;
            }
          }
          //mark neighbor to parent[neighbor](both ways)
          int local_parent_neighbor = get_value(g->map, parents[neighbor]);
          for(int k = g->out_degree_list[neighbor]; k < g->out_degree_list[neighbor+1]; k++){
            if(g->out_edges[k] == local_parent_neighbor){
              printf("Task %d: marking edge from %d to %d as visited\n",procid,neighbor,g->out_edges[k]);
              visited_edges[k] = 1;
            }
          }
          for(int k = g->out_degree_list[local_parent_neighbor]; k < g->out_degree_list[local_parent_neighbor+1]; k++){
            if(g->out_edges[k] == neighbor){
              printf("Task %d: marking edge from %d to %d as visited\n",procid,local_parent_neighbor,g->out_edges[k]);
              visited_edges[k] = 1;
            }
          }
          
          q.push(global_current);
          q.push(global_neighbor);
          q.push(levels[i]);
          q.push(levels[neighbor]);
          q.push(parents[i]);
          q.push(parents[neighbor]);
          if(parents[i] < g->n_local){
            q.push(procid);
            printf("Task %d: Proc1 = %d\n",procid, procid);
          } else {
            q.push(g->ghost_tasks[parents[i] - g->n_local]);
            printf("Task %d: Proc1 = %d\n",procid, g->ghost_tasks[parents[i]-g->n_local]);
          }
          if(parents[neighbor] < g->n_local){
            q.push(procid);
            printf("Task %d: Proc2 = %d\n",procid, procid);
            
          } else {
            q.push(g->ghost_tasks[parents[neighbor] - g->n_local]);
            printf("Task %d: Proc2 = %d\n",procid, g->ghost_tasks[parents[neighbor]-g->n_local]);
          }
        }
      }
    }
  }
}

void lca_traversal(dist_graph_t* g, std::queue<int> &queue, std::queue<int> &send, uint64_t* parents, uint64_t* levels, uint64_t* flags, int* visited_edges){
  //every vertex id is global in this function, when used, we need to translate to local.
  while(!queue.empty()){
    int vertex1 = queue.front();
    queue.pop();
    int vertex2 = queue.front();
    queue.pop();
    int level1 = queue.front();
    queue.pop();
    int level2 = queue.front();
    queue.pop();
    int parent1 = queue.front();
    queue.pop();
    int parent2 = queue.front();
    queue.pop();
    int proc1 = queue.front();
    queue.pop();
    int proc2 = queue.front();
    queue.pop();
    
    printf("Task %d: v1: %d, v2: %d, l1: %d, l2: %d, p1: %d, p2: %d, P1: %d, P2: %d\n",procid,vertex1, vertex2, level1, level2, parent1, parent2, proc1, proc2);    

    if(parent1 == parent2){
      flags[get_value(g->map,parent1)] = 1;
      printf("vertex %d is an LCA\n",parent1);
      if(proc1 != procid || proc2 != procid){
        send.push(vertex1); send.push(vertex2);
        send.push(level1);  send.push(level2);
        send.push(parent1); send.push(parent2);
        send.push(proc1);   send.push(proc2);
      }
      continue;
    }
    //if the current processor can do work on this entry
    if((proc1 == procid && level1 >= level2)||(proc2 == procid && level2 >= level1)){
      //do that work, and put the entry back on the queue.
      if(level1 >= level2){ //advance the first entry
        int local_vtx1 = get_value(g->map, vertex1);
        int local_parent1 = get_value(g->map, parent1);
	//this local_vtx1 to local_parent1 edge was already marked, mark the new edge as visited
        //vertex1 should equal parent1 (both are global IDs)
        vertex1 = parent1;
        //parent1 should be local_parent1's parent (which is global)
        parent1 = parents[local_parent1];
        //if the new parent is ghosted, the owning processor should change.
        //this is the case so we have consistent edge visitation information
        // owned vertex => ghosted parent gets marked, ghosted child => owned parent gets marked
        int local_grandparent = get_value(g->map,parent1);
        //now, mark this new edge as visited
        for(int i = g->out_degree_list[local_parent1]; i < g->out_degree_list[local_parent1+1]; i++){
          if(g->out_edges[i] == local_grandparent){
            printf("Task %d: marking edge from %d to %d as visited\n",procid,local_parent1, local_grandparent);
            visited_edges[i] = 1;
          }
        }
        for(int i = g->out_degree_list[local_grandparent]; i< g->out_degree_list[local_grandparent+1]; i++){
          if(g->out_edges[i] == local_parent1){
            printf("Task %d: marking edge from %d to %d as visited\n",procid, local_grandparent, local_parent1);
            visited_edges[i] = 1;
          }
        }
        if(local_grandparent >= g->n_local){
          proc1 = g->ghost_tasks[local_grandparent - g->n_local];
        }
        level1--; 
      } else { //advance the second entry.
        int local_vtx2 = get_value(g->map, vertex2);
        int local_parent2 = get_value(g->map, parent2);
	//mark the new edge as visited (local_parent2 to local_grandparent)       
 
        //vertex2 should equal parent2 (both are global IDs)
        vertex2 = parent2;
        //parent2 should be local_parent2's parent (which is a global ID)
        parent2 = parents[local_parent2];
        //if the new parent is ghosted, the owning processor should change
        int local_grandparent = get_value(g->map, parent2);
        for(int i = g->out_degree_list[local_parent2]; i < g->out_degree_list[local_parent2 + 1]; i++){
          if(g->out_edges[i] == local_grandparent){
            printf("Task %d: marking edge from %d to %d as visited\n",procid, local_parent2, local_grandparent);
            visited_edges[i] = 1;
          }
        }
        for(int i  = g->out_degree_list[local_grandparent]; i < g->out_degree_list[local_grandparent+1]; i++){
          if(g->out_edges[i] == local_parent2){
            printf("Task %d: marking edge from %d to %d as visited\n",procid, local_grandparent, local_parent2);
            visited_edges[i] = 1;
          }
        }
        
        if(local_grandparent >= g->n_local){
          proc2 = g->ghost_tasks[local_grandparent - g->n_local];
        }
        level2--;
      }
      //put the values back on the queue
      queue.push(vertex1); queue.push(vertex2);
      queue.push(level1);  queue.push(level2);
      queue.push(parent1); queue.push(parent2);
      queue.push(proc1);   queue.push(proc2);
    } else {
      // send the entry to one of the processors.
      send.push(vertex1); send.push(vertex2);
      send.push(level1);  send.push(level2);
      send.push(parent1); send.push(parent2);
      send.push(proc1);   send.push(proc2);
    }
  }  
}

void communicate(dist_graph_t* g, std::queue<int> &send, std::queue<int> &queue, int* visited_edges){
  //figure out how many entries are being sent to each processor
  uint64_t* sendbuf = new uint64_t[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendbuf[i] = 0;
  }
  
  std::queue<int> procsqueue;
  
  for(int i = 0; i < send.size()/8; i++){
    int vertex1 = send.front();
    send.pop();
    int vertex2 = send.front();
    send.pop();
    int level1 = send.front();
    send.pop();
    int level2 = send.front();
    send.pop();
    int parent1 = send.front();
    send.pop();
    int parent2 = send.front();
    send.pop();
    int proc1 = send.front();
    send.pop();
    int proc2 = send.front();
    send.pop();
    printf("Task %d sending: vertex1: %d, vertex2: %d, level1: %d, level2: %d, parent1: %d, parent2: %d, proc1: %d, proc2: %d\n", procid,vertex1, vertex2,level1,level2,parent1,parent2,proc1, proc2);
    
    if(proc1 != procid){ //send to proc1
      sendbuf[proc1]++;
      procsqueue.push(proc1);
    } else if(proc2 != procid){ //send to proc2
      sendbuf[proc2]++;
      procsqueue.push(proc2);
    }
    
    send.push(vertex1);
    send.push(vertex2);
    send.push(level1);
    send.push(level2);
    send.push(parent1);
    send.push(parent2);
    send.push(proc1);
    send.push(proc2);
  } 
  printf("Task %d Sendbuf: ",procid);
  for(int i = 0; i < nprocs; i++){
    printf("%d ",sendbuf[i]);
  }
  printf("\n");
  //send the counts using alltoall
  uint64_t* recvbuf = new uint64_t[nprocs];
  int status = MPI_Alltoall(sendbuf, nprocs, MPI_INT,recvbuf, nprocs, MPI_INT, MPI_COMM_WORLD);
  printf("Task %d Recvbuf: ",procid);
  for(int i = 0; i < nprocs; i++){
    printf("%d ",recvbuf[i]);
  }  
  printf("\n");

  int* sdispls = new int[nprocs];
  sdispls[0] = 0;
  int* rdispls = new int[nprocs];
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendbuf[i-1]*8;
    rdispls[i] = rdispls[i-1] + recvbuf[i-1]*8;
  }
  int sendsize = 0;
  int recvsize = 0;
  int* sentcount = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendsize += sendbuf[i]*8;
    recvsize += recvbuf[i]*8;
    sentcount[i] = 0;
  }
  int* final_sendbuf = new int[sendsize];
  int* final_recvbuf = new int[recvsize];
  while(!send.empty()){
    int proc_to_send = procsqueue.front();
    procsqueue.pop();
    int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send]*8;
    sentcount[proc_to_send]++;
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//vertex1
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//vertex2
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//level1
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//level2
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//parent1
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//parent2
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//proc1
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//proc2
  }

  int* sendcounts = new int[nprocs];
  int* recvcounts = new int[nprocs];
  for(int i = 0 ; i < nprocs; i++){
    sendcounts[i] = sendbuf[i]*8;
    recvcounts[i] = recvbuf[i]*8;
  }
  
  printf("Task %d sendbuf: ",procid);
  for(int i = 0; i < sendsize; i+=8){
    printf("%d,%d;%d,%d;%d,%d;%d,%d; ",final_sendbuf[i],final_sendbuf[i+1],final_sendbuf[i+2],final_sendbuf[i+3],final_sendbuf[i+4],final_sendbuf[i+5],final_sendbuf[i+6],final_sendbuf[i+7]);
  }
  printf("\n");
  
  printf("Task %d sdispls: ",procid);
  for(int i = 0; i < nprocs; i++){
    printf("%d ",sdispls[i]);
  }
  printf("\n");  

  printf("Task %d recvcounts: ",procid);
  for(int i = 0; i < nprocs; i++){
    printf("%d ",recvcounts[i]);
  }
  printf("\n");  
  printf("Task %d rdispls: ",procid);
  for(int i = 0; i < nprocs; i++){
    printf("%d ",rdispls[i]);
  }
  printf("\n");

  //using the counts, put each entry in an array to send
  //calculate displacements and everything else needed for the alltoallv.
  status = MPI_Alltoallv(final_sendbuf, sendcounts, sdispls, MPI_INT, final_recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  
  for(int i = 0; i < recvsize; i+=8){
    uint64_t vtx1 = final_recvbuf[i];
    uint64_t vtx2 = final_recvbuf[i+1];
    uint64_t level1 = final_recvbuf[i+2];
    uint64_t level2 = final_recvbuf[i+3];
    uint64_t parent1 = final_recvbuf[i+4];
    uint64_t parent2 = final_recvbuf[i+5];
    uint64_t proc1 = final_recvbuf[i+6];
    uint64_t proc2 = final_recvbuf[i+7];
    printf("Task %d received entry: vtx1=%d, vtx2=%d, lvl1=%d, lvl2=%d, prnt1=%d, prnt2=%d, proc1=%d, proc2=%d\n",procid, vtx1,vtx2,level1,level2,parent1,parent2,proc1,proc2);
  }
  // take the entries in final_recvbuf and push them on the regular queue.
  for(int i = 0; i < recvsize; i+=8){
    int vertex1 = final_recvbuf[i];
    int vertex2 = final_recvbuf[i+1];
    int level1 = final_recvbuf[i+2];
    int level2 = final_recvbuf[i+3];
    int parent1 = final_recvbuf[i+4];
    int parent2 = final_recvbuf[i+5];
    int proc1 = final_recvbuf[i+6];
    int proc2 = final_recvbuf[i+7];
    
    //mark the vertex1 to parent1 edges as visited (both ways?)
    int localvertex1 = get_value(g->map, vertex1);
    int localparent1 = get_value(g->map, parent1);
    for(int j = g->out_degree_list[localvertex1]; j < g->out_degree_list[localvertex1+1]; j++){
      if(g->out_edges[j] == localparent1){
        printf("Task %d: marking edge from %d to %d\n",procid,localvertex1,g->out_edges[j]);
        visited_edges[j] = 1;
      }
    }
    for(int j = g->out_degree_list[localparent1]; j < g->out_degree_list[localparent1+1]; j++){
      if(g->out_edges[j] == localvertex1){
        printf("Task %d: marking edge from %d to %d\n",procid,localparent1,g->out_edges[j]);
        visited_edges[j] = 1;
      }
    }
    //mark the vertex2 to parent2 edges as visited
    int localvertex2 = get_value(g->map, vertex2);
    int localparent2 = get_value(g->map, parent2);
    for(int j = g->out_degree_list[localvertex2]; j < g->out_degree_list[localvertex2+1]; j++){
      if(g->out_edges[j] == localparent2){
        printf("Task %d: marking edge from %d to %d\n",procid,localvertex2,g->out_edges[j]);
        visited_edges[j] = 1;
      }
    }
    for(int j = g->out_degree_list[localparent2]; j < g->out_degree_list[localparent2+1]; j++){
      if(g->out_edges[j] == localvertex2){
        printf("Task %d: marking edge from %d to %d\n",procid,localparent2,g->out_edges[j]);
        visited_edges[j] = 1;
      }
    }

    queue.push(vertex1);
    queue.push(vertex2);
    queue.push(level1); 
    queue.push(level2);
    queue.push(parent1);
    queue.push(parent2);
    queue.push(proc1);
    queue.push(proc2);
  }  

}

void art_pt_heuristic(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, 
                      uint64_t* parents, uint64_t* levels, uint64_t* art_pt_flags) {
  //Initialize the queue q->queue with nontree edges.
  std::queue<int> lca_data;
  int* visited_edges = new int[g->m_local];
  init_queue_nontree(g,lca_data,parents,levels,visited_edges);
  printf("Task %d found %d nontree edges\n",procid, lca_data.size()/6);
  //do LCA traversals incrementally, communicating in batches
    //also need to keep track of the edges used in the traversal, to flag both ends of a bridge.
  std::queue<int> send_queue;
  int all_done = 0;
  while(!all_done){
    lca_traversal(g,lca_data,send_queue,parents,levels,art_pt_flags,visited_edges);
    printf("Task %d attempting to send %d entries\n",procid,send_queue.size()/6);
    communicate(g,send_queue,lca_data,visited_edges);
    int local_done = lca_data.empty() && send_queue.empty();
    MPI_Allreduce(&local_done, &all_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
  //any endpoint of an unvisited edge should be marked as a potential articulation point
  for(int i = 0; i < g->n_local; i++){
    //printf("Task %d: node %d has degree %d\n",procid, i, g->out_degree_list[i+1]-g->out_degree_list[i]);
    //else printf("Task %d: node %d has degree %d\n",procid,i,g->ghost_degrees[i-g->n_local+1]-g->ghost_degrees[i-g->n_local]);
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      if(visited_edges[j] == 0){
        printf("Task %d: edge from %d to %d is a bridge\n",procid, i, g->out_edges[j]);
      }
    }
  }
}

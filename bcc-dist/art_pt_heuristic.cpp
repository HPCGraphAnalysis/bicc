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

void init_queue_nontree(dist_graph_t* g, std::queue<int> &q, uint64_t* parents,uint64_t* levels){
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
          q.push(i);
          q.push(neighbor);
          q.push(levels[i]);
          q.push(levels[neighbor]);
          q.push(get_value(g->map,parents[i]));
          q.push(get_value(g->map,parents[neighbor]));
        }
      }
    }
  }
}

void lca_traversal(dist_graph_t* g, std::queue<int> &queue, std::queue<int> &send, uint64_t* parents, uint64_t* levels){
  int count = 0;
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
    if(parent1 == parent2) {
      //if the vertex is ghosted, send it to the owning processor.
      //also, mark it as an LCA for this processor.
      if(parent1 >= g->n_local){
        printf("Task %d sending completed traversal(p1==p2)\n",procid);
        send.push(vertex1);
        send.push(vertex2);
        send.push(level1);
        send.push(level2);
        send.push(parent1);
        send.push(parent2);
        continue;
      } else {
        printf("vertex %d is an LCA\n",g->local_unmap[parent1]);
        continue;
      }
    }
    if(vertex1 == vertex2){
      //if the vertex is ghosted, send it to the owning processor.
      //also, mark it as an LCA for this processor.
      if(vertex1 >= g->n_local){
        printf("Task %d sending completed traversal(v1==v2)\n",procid);
        send.push(vertex1);
        send.push(vertex2);
        send.push(level1);
        send.push(level2);
        send.push(parent1);
        send.push(parent2);
        continue;
      } else {
        printf("vertex %d is an LCA\n", g->local_unmap[vertex1]);
        continue;
      }
    }
    //if(parent1 >= g->n_local) printf("Parent1 is a ghosted vertex\n");
    //if(parent2 >= g->n_local) printf("Parent2 is a ghosted vertex\n");
    printf("Task %d: vertex1: %d, vertex2: %d, level1: %d, level2: %d, parent1: %d, parent2: %d\n", procid,vertex1, vertex2,level1,level2,parent1,parent2);
    //if(vertex1 == 4 && vertex2 == 2) continue;
    
    //if(count==2) break;
    if(level1 > level2){
      //check out vertex1 and parent1, see if vertex1 is locally owned
      if(vertex1 < g->n_local){
        vertex1 = parent1;
        parent1 = get_value(g->map,parents[vertex1]);
        level1 = levels[vertex1];
        queue.push(vertex1);
        queue.push(vertex2);
        queue.push(level1);
        queue.push(level2);
        queue.push(parent1);
        queue.push(parent2);
        //push everything back on the queue
      } else {
        //put the entry on the send queue.
        send.push(vertex1);
        send.push(vertex2);
        send.push(level1);
        send.push(level2);
        send.push(parent1);
        send.push(parent2);
      }
    } else if(level1 < level2){
      //check out vertex2 and parent2, see if vertex2 is locally owned
      if(vertex2 < g->n_local){
        vertex2 = parent2;
        parent2 = get_value(g->map,parents[vertex2]);
        level2 = levels[vertex2];
        queue.push(vertex1);
        queue.push(vertex2);
        queue.push(level1);
        queue.push(level2);
        queue.push(parent1);
        queue.push(parent2);
        //push everything back on the queue
      } else {
        //put the entry on the send queue
        send.push(vertex1);
        send.push(vertex2);
        send.push(level1);
        send.push(level2);
        send.push(parent1);
        send.push(parent2);
        
      }
    } else {
      //see if either vertex1 or vertex2 is locally owned
      if(vertex1 < g->n_local){
        vertex1 = parent1;
        parent1 = get_value(g->map,parents[vertex1]);
        level1 = levels[vertex1];
        //push everything back on the queue
        queue.push(vertex1);
        queue.push(vertex2);
        queue.push(level1);
        queue.push(level2);
        queue.push(parent1);
        queue.push(parent2);
      } else if(vertex2 < g->n_local){
        vertex2 = parent2;
        parent2 = get_value(g->map,parents[vertex2]);
        level2 = levels[vertex2];
        //push everything back on the queue
        queue.push(vertex1);
        queue.push(vertex2);
        queue.push(level1);
        queue.push(level2);
        queue.push(parent1);
        queue.push(parent2);
      } else {
        //put entry on the send queue
        send.push(vertex1);
        send.push(vertex2);
        send.push(level1);
        send.push(level2);
        send.push(parent1);
        send.push(parent2);
      }
    }
    count++;
  }
}

void communicate(dist_graph_t* g, std::queue<int> &send, std::queue<int> &queue){
  //figure out how many entries are being sent to each processor
  uint64_t* sendbuf = new uint64_t[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendbuf[i] = 0;
  }
  
  std::queue<int> procsqueue;
  
  for(int i = 0; i < send.size()/6; i++){
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
    printf("Task %d sending: vertex1: %d, vertex2: %d, level1: %d, level2: %d, parent1: %d, parent2: %d\n", procid,vertex1, vertex2,level1,level2,parent1,parent2);
    
    if(vertex1 == vertex2 || parent1 == parent2){
      //printf("Task %d attempting to send a completed traversal\n",procid);
      int sendto = -1;
      if(vertex1 == vertex2) sendto = g->ghost_tasks[vertex1-g->n_local];
      else if(parent1 == parent2) sendto = g->ghost_tasks[parent1-g->n_local];
      sendbuf[sendto]++;
      procsqueue.push(sendto);
    }else if(level1 > level2){
      //vertex1 is a ghost, push the entry over to the owning processor
      int sendto = g->ghost_tasks[vertex1-g->n_local];
      sendbuf[sendto]++;
      procsqueue.push(sendto);
    } else if(level1 < level2) {
      //vertex2 is a ghost, push the entry!
      int sendto = g->ghost_tasks[vertex2-g->n_local];
      sendbuf[sendto]++;
      procsqueue.push(sendto);
    } else {
      if(vertex1 >= g->n_local){
        //vertex1 is ghosted
        int sendto = g->ghost_tasks[vertex1-g->n_local];
        sendbuf[sendto]++;
        procsqueue.push(sendto);
      } else {
        //this may not ever get reached, I'm not sure...
        //vertex2 must be ghosted, since vertex1 is not, and this must be sent somewhere.
        int sendto = g->ghost_tasks[vertex2-g->n_local];
        sendbuf[sendto]++;
        procsqueue.push(sendto);
      }
    }
    //switch to global IDs
    send.push(g->local_unmap[vertex1]);
    send.push(g->local_unmap[vertex2]);
    send.push(level1);
    send.push(level2);
    if(parent1 < g->n_local) send.push(g->local_unmap[parent1]);
    else send.push(g->ghost_unmap[parent1-g->n_local]);
    if(parent2 < g->n_local) send.push(g->local_unmap[parent2]);
    else send.push(g->ghost_unmap[parent2-g->n_local]);
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
    sdispls[i] = sdispls[i-1] + sendbuf[i-1]*6;
    rdispls[i] = rdispls[i-1] + recvbuf[i-1]*6;
  }
  int sendsize = 0;
  int recvsize = 0;
  int* sentcount = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendsize += sendbuf[i]*6;
    recvsize += recvbuf[i]*6;
    sentcount[i] = 0;
  }
  int* final_sendbuf = new int[sendsize];
  int* final_recvbuf = new int[recvsize];
  while(!send.empty()){
    int proc_to_send = procsqueue.front();
    procsqueue.pop();
    int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send]*6;
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
  }

  int* sendcounts = new int[nprocs];
  int* recvcounts = new int[nprocs];
  for(int i = 0 ; i < nprocs; i++){
    sendcounts[i] = sendbuf[i]*6;
    recvcounts[i] = recvbuf[i]*6;
  }
  
  printf("Task %d sendbuf: ",procid);
  for(int i = 0; i < sendsize; i+=6){
    printf("%d,%d;%d,%d;%d,%d; ",final_sendbuf[i],final_sendbuf[i+1],final_sendbuf[i+2],final_sendbuf[i+3],final_sendbuf[i+4],final_sendbuf[i+5]);
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
  
  for(int i = 0; i < recvsize; i+=6){
    uint64_t vtx1 = final_recvbuf[i];
    uint64_t vtx2 = final_recvbuf[i+1];
    uint64_t level1 = final_recvbuf[i+2];
    uint64_t level2 = final_recvbuf[i+3];
    uint64_t parent1 = final_recvbuf[i+4];
    uint64_t parent2 = final_recvbuf[i+5];
    printf("Task %d received entry: vtx1=%d, vtx2=%d, lvl1=%d, lvl2=%d, prnt1=%d, prnt2=%d\n",procid, vtx1,vtx2,level1,level2,parent1,parent2);
  }
  // take the entries in final_recvbuf and translate the global ids to local, and push them on the regular queue.
  for(int i = 0; i < recvsize; i+=6){
    int vertex1 = get_value(g->map,final_recvbuf[i]);
    int vertex2 = get_value(g->map,final_recvbuf[i+1]);
    int level1 = final_recvbuf[i+2];
    int level2 = final_recvbuf[i+3];
    int parent1 = get_value(g->map, final_recvbuf[i+4]);
    int parent2 = get_value(g->map, final_recvbuf[i+5]);
    queue.push(vertex1);
    queue.push(vertex2);
    queue.push(level1); 
    queue.push(level2);
    queue.push(parent1);
    queue.push(parent2);
  }  

}

void art_pt_heuristic(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, 
                      uint64_t* parents, uint64_t* levels, uint64_t* art_pt_flags) {
  //Initialize the queue q->queue with nontree edges.
  std::queue<int> lca_data;
  init_queue_nontree(g,lca_data,parents,levels);
  printf("Task %d found %d nontree edges\n",procid, lca_data.size()/6);
  //do LCA traversals incrementally, communicating in batches
    //also need to keep track of the edges used in the traversal, to flag both ends of a bridge.
  std::queue<int> send_queue;
  int all_done = 0;
  while(!all_done){
    lca_traversal(g,lca_data,send_queue,parents,levels);
    printf("Task %d attempting to send %d entries\n",procid,send_queue.size()/6);
    communicate(g,send_queue,lca_data);
    int local_done = lca_data.empty() && send_queue.empty();
    MPI_Allreduce(&local_done, &all_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
  //store appropriate values in art_pt_flags
  
}

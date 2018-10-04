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
        if(global_current < global_neighbor){
          if(levels[i] <= levels[neighbor]){
            //if neighbor is owned, this processor gets the entry
            if(neighbor < g->n_local){
              printf("Task %d: nontree edge found between %d and %d\n",procid,global_current,global_neighbor);
              q.push(global_current);
              q.push(global_neighbor);
              q.push(levels[i]);
              q.push(levels[neighbor]);
              if(i < g->n_local) q.push(procid);
              else q.push(g->ghost_tasks[i-g->n_local]);
              q.push(procid);
              //mark this nontree edge as visited
              //we already know i is a local vertex
              for(int k = g->out_degree_list[i]; k < g->out_degree_list[i+1]; k++){
                if(g->out_edges[k] == neighbor){
                  visited_edges[k] = 1;
                }
              }
              if(neighbor < g->n_local){
                for(int k = g->out_degree_list[neighbor]; k < g->out_degree_list[neighbor+1]; k++){
                  if(g->out_edges[k] == i){
                    visited_edges[k] = 1;
                  }
                }
              }
              
            }
          } else {
            //we already know i is owned, i goes from 0 to g->n_local
            printf("Task %d: nontree edge found between %d and %d\n",procid,global_current,global_neighbor);
            q.push(global_current);
            q.push(global_neighbor);
            q.push(levels[i]);
            q.push(levels[neighbor]);
            q.push(procid);
            if(neighbor < g->n_local) q.push(procid);
            else q.push(g->ghost_tasks[neighbor-g->n_local]);
            //mark this nontree edge as visited
            //we already know i is a local vertex
            for(int k = g->out_degree_list[i]; k < g->out_degree_list[i+1]; k++){
              if(g->out_edges[k] == neighbor){
                visited_edges[k] = 1;
              }
            }
            if(neighbor < g->n_local){
              for(int k = g->out_degree_list[neighbor]; k < g->out_degree_list[neighbor+1]; k++){
                if(g->out_edges[k] == i){
                  visited_edges[k] = 1;
                }
              }
            }
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
    int task1 = queue.front();
    queue.pop();
    int task2 = queue.front();
    queue.pop();

    printf("Task %d: v1: %d, v2: %d, l1: %d, l2: %d, t1: %d, t2: %d\n",procid,vertex1,vertex2,level1,level2,task1,task2);
    int local_vertex1 = get_value(g->map,vertex1);
    int local_vertex2 = get_value(g->map,vertex2);
    if(local_vertex1 >= 0 && local_vertex2 >= 0){
      //make sure that both vertices are at least ghosted, so we have parent info for both.
      //if not, we can't mark an LCA.
      if(local_vertex1 == local_vertex2){
        //this LCA may be ghosted, send it to the other processor that has it ghosted. 
        printf("Task %d: vertex %d is an LCA\n",procid, local_vertex1);
        if(task1 != procid){
          printf("Task %d: need to send %d,%d;%d,%d;%d,%d; entry to Task %d\n",procid,local_vertex1,local_vertex2,level1,level2,task1,task1,task1);
          task2 = task1;
          send.push(local_vertex1); send.push(local_vertex2);
          send.push(level1);        send.push(level2);
          send.push(task1);                  send.push(task2);
        }
        if(task2 != procid){
          printf("Task %d: need to send %d,%d;%d,%d;%d,%d; entry to Task %d\n",procid,local_vertex1,local_vertex2,level1,level2,task2,task2,task2);
          task1 = task2;
          send.push(local_vertex1); send.push(local_vertex2);
          send.push(level1);        send.push(level2);
          send.push(task1);         send.push(task2);
        }
        continue;
      }
    }

    //if the local processor can do more work
    if((level1 >= level2 && procid == task1) || (level2 >= level1 && procid == task2)){
      if(level1 >= level2 && procid == task1){
        //int local_vertex1 = get_value(g->map,vertex1);
        int local_parent1 = get_value(g->map,parents[local_vertex1]);
        //mark vertex1 to parent[vertex1] as visited
        if(local_vertex1 < g->n_local){
          //we can mark the edge from vertex1 to parents[vertex1] as visited safely
          for(int i = g->out_degree_list[local_vertex1]; i < g->out_degree_list[local_vertex1+1]; i++){
            if(local_parent1 == g->out_edges[i]){
              visited_edges[i] = 1;
            }
          }
        }
        if(local_parent1 < g->n_local){
          //we can mark the edge from parents[vertex1] to vertex1 as visited safely
          for(int i = g->out_degree_list[local_parent1]; i < g->out_degree_list[local_parent1+1]; i++){
            if(local_vertex1 == g->out_edges[i]){
              visited_edges[i] = 1;
            }
          }
        }
        //either advance the entry on this processor, or send to the processor that owns parents[vertex1]
        if(local_parent1 < g->n_local){
          vertex1 = parents[local_vertex1];
          level1--;
          //the parent vertex is locally owned,
          //so it is owned by the current processor (no need to update task1)
        } else {
          //vertex1 = parents[local_vertex1];
          //level1--;
          //update the owning task for this entry, don't change it.
          task1 = g->ghost_tasks[local_parent1-g->n_local];
        }
      } else {
        //do work on vertex2
        //int local_vertex2 = get_value(g->map, vertex2);
        int local_parent2 = get_value(g->map, parents[local_vertex2]);
        //mark vertex2 to parent[vertex2] as visited
        if(local_vertex2 < g->n_local){
          for(int i = g->out_degree_list[local_vertex2]; i < g->out_degree_list[local_vertex2+1]; i++){
            if(local_parent2 == g->out_edges[i]){
              visited_edges[i] = 1;
            }
          }
        }
        if(local_parent2 < g->n_local){
          for(int i = g->out_degree_list[local_parent2]; i < g->out_degree_list[local_parent2+1]; i++){
            if(local_vertex2 == g->out_edges[i]){
              visited_edges[i] = 1;
            }
          }
        }
        //either advance the entry on this processor, or send to the processor that owns parents[vertex2]
        if(local_parent2 < g->n_local){
          vertex2 = parents[local_vertex2];
          level2--;
        } else {
          //vertex2 = parents[local_vertex2];
          //level2--;
          //update the owning task for this entry, don't change anything else.
          task2 = g->ghost_tasks[local_parent2-g->n_local];
        }
      }
      queue.push(vertex1); queue.push(vertex2);
      queue.push(level1);  queue.push(level2);
      queue.push(task1);   queue.push(task2);
    } else {
      //put the entry on the send queue.
      send.push(vertex1); send.push(vertex2);
      send.push(level1);  send.push(level2);
      send.push(task1);   send.push(task2);
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
  
  for(int i = 0; i < send.size()/6; i++){
    int vertex1 = send.front();
    send.pop();
    int vertex2 = send.front();
    send.pop();
    int level1 = send.front();
    send.pop();
    int level2 = send.front();
    send.pop();
    int proc1 = send.front();
    send.pop();
    int proc2 = send.front();
    send.pop();
    printf("Task %d sending: vertex1: %d, vertex2: %d, level1: %d, level2: %d, proc1: %d, proc2: %d\n", procid,vertex1, vertex2,level1,level2,proc1, proc2);
    
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
    send.pop();//proc1
    final_sendbuf[sendbufidx++] = send.front();
    send.pop();//proc2
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
    uint64_t proc1 = final_recvbuf[i+4];
    uint64_t proc2 = final_recvbuf[i+5];
    printf("Task %d received entry: vtx1=%d, vtx2=%d, lvl1=%d, lvl2=%d, proc1=%d, proc2=%d\n",procid, vtx1,vtx2,level1,level2,proc1,proc2);
  }
  // take the entries in final_recvbuf and push them on the regular queue.
  for(int i = 0; i < recvsize; i+=6){
    int vertex1 = final_recvbuf[i];
    int vertex2 = final_recvbuf[i+1];
    int level1 = final_recvbuf[i+2];
    int level2 = final_recvbuf[i+3];
    int proc1 = final_recvbuf[i+4];
    int proc2 = final_recvbuf[i+5];
    
    queue.push(vertex1);
    queue.push(vertex2);
    queue.push(level1); 
    queue.push(level2);
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
        int global_j = 0;
        if(g->out_edges[j] < g->n_local) global_j = g->local_unmap[g->out_edges[j]];
        else global_j = g->ghost_unmap[g->out_edges[j] - g->n_local];
        printf("Task %d: edge from %d to %d is a bridge\n",procid, g->local_unmap[i], global_j);
      }
    }
  }
}

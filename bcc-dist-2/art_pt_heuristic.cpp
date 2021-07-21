/* insert copyright stuff here*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <queue>
#include <iostream>
#include "dist_graph.h"
//#include "art_pt_heuristic_comms.h" <-This may be added in the future
#include "art_pt_heuristic.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;

void init_queue_nontree(dist_graph_t* g, std::queue<int> &q, uint64_t* parents,uint64_t* levels, int* visited_edges){
  std::cout<<"Computing nontree edges\n";
  for(int i = 0; i < g->n_local; i++){
    //go through the local nodes, and look through all connections that are not parent-child
    int out_degree = out_degree(g,i);
    uint64_t* outs = out_vertices(g,i);
    for(int j = 0; j < out_degree; j++){
      uint64_t neighbor = outs[j];
      uint64_t global_neighbor = neighbor;
      if(neighbor >= g->n_local) global_neighbor = g->ghost_unmap[global_neighbor-g->n_local];
      else global_neighbor = g->local_unmap[global_neighbor];
      uint64_t global_current = g->local_unmap[i];
      printf("Checking edge between %d (parent: %d) and %d (parent: %d)\n",global_current,parents[i],global_neighbor,parents[neighbor]);
      if(parents[neighbor] != global_current && parents[i] != global_neighbor){
        //if the edge is partly owned by the current processor, the neighbor will be the ghosted vertex
        if((global_current < global_neighbor) || (neighbor >= g->n_local)){		
          if(levels[i] <= levels[neighbor]){
            //if neighbor is owned, this processor gets the entry
            if(neighbor < g->n_local){
              printf("Task %d: nontree edge found between %d and %d\n",procid,global_current,global_neighbor);
              q.push(global_current);
              q.push(global_neighbor);
              q.push(levels[i]);
              q.push(levels[neighbor]);
              q.push(procid);
              q.push(procid);
              //mark this nontree edge as visited
              //we already know i is a local vertex
            } else {
              //neighbor is ghosted
	      if(levels[i] >= levels[neighbor]){
	        //if the neighbor is at a lower numerical level (so, higher up the tree) than i, this process does not own the traversal to start
		
		//the only edges that make it here have levels[i] == levels[neighbor], so distinguish based on GID.
		if(global_current < global_neighbor){
                  printf("Task %d: nontree edge found between %d and %d\n",procid,global_current,global_neighbor);
                  q.push(global_current);
                  q.push(global_neighbor);
                  q.push(levels[i]);
                  q.push(levels[neighbor]);
                  q.push(procid);
                  q.push(g->ghost_tasks[neighbor-g->n_local]);
		}
	      }

              
	    }
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
                if(procid == 1){
                  int from = g->local_unmap[i];
                  int to = 0;
                  if(neighbor < g->n_local) to = g->local_unmap[neighbor];
                  else to = g->ghost_unmap[neighbor];
                  //printf("Task 1 marked edge from %d to %d as visited\n",from,to);
                }
                visited_edges[k] = 1;
              }
            }
            if(neighbor < g->n_local){
              for(int k = g->out_degree_list[neighbor]; k < g->out_degree_list[neighbor+1]; k++){
                if(g->out_edges[k] == i){
                  if(procid == 1){
                    int from = g->local_unmap[i];
                    int to = 0;
                    if(neighbor < g->n_local) to = g->local_unmap[neighbor];
                    else to = g->ghost_unmap[neighbor];
                    //printf("Task 1 marked edge from %d to %d as visited\n",from,to);
                  }
                  visited_edges[k] = 1;
                }
              }
            }
          }
        } else {
          //mark this nontree edge as visited (this is on the processor that doesn't get the entry to start)
          /*for(int k = g->out_degree_list[i]; k < g->out_degree_list[i+1]; k++){
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
          }*/
        }
      }
    }
  }
  std::cout<<"Finished computing nontree edges\n";
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
        //printf("Task %d: vertex %d is an LCA\n",procid, local_vertex1);
        flags[local_vertex1] = 1;
        if(task1 != procid){
          //printf("Task %d: need to send %d,%d;%d,%d;%d,%d; entry to Task %d\n",procid,local_vertex1,local_vertex2,level1,level2,task1,task1,task1);
          task2 = task1;
          send.push(local_vertex1); send.push(local_vertex2);
          send.push(level1);        send.push(level2);
          send.push(task1);         send.push(task2);
        }
        if(task2 != procid){
          //printf("Task %d: need to send %d,%d;%d,%d;%d,%d; entry to Task %d\n",procid,local_vertex1,local_vertex2,level1,level2,task2,task2,task2);
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
              if(procid==1){
                //printf("Task 1 marking edge from %d to %d as visited\n",vertex1,parents[local_vertex1]);
              }
              visited_edges[i] = 1;
            }
          }
        }
        if(local_parent1 < g->n_local){
          //we can mark the edge from parents[vertex1] to vertex1 as visited safely
          for(int i = g->out_degree_list[local_parent1]; i < g->out_degree_list[local_parent1+1]; i++){
            if(local_vertex1 == g->out_edges[i]){
              if(procid==1){
                //printf("Task 1 marking edge from %d to %d as visited\n",parents[local_vertex1],vertex1);
              }
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
              if(procid==1){
                //printf("Task 1 marking edge from %d to %d as visited\n",vertex2,parents[local_vertex2]);
              }
              visited_edges[i] = 1;
            }
          }
        }
        if(local_parent2 < g->n_local){
          for(int i = g->out_degree_list[local_parent2]; i < g->out_degree_list[local_parent2+1]; i++){
            if(local_vertex2 == g->out_edges[i]){
              if(procid==1){
                //printf("Task 1 marking edge from %d to %d as visited\n",parents[local_vertex2],vertex2);
              }
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
  std::cout<<"Rank "<<procid<<" communicating LCA Traversal\n";
  //figure out how many entries are being sent to each processor
  int* sendbuf = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendbuf[i] = 0;
  }
  
  std::queue<int> procsqueue;
  std::cout<<"Rank "<<procid<<" setting up sendnts\n";
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
  int* recvbuf = new int[nprocs];
  for(int i = 0; i < nprocs; i++) recvbuf[i] = 0;
  
  int status = MPI_Alltoall(sendbuf, 1, MPI_INT,recvbuf, 1, MPI_INT, MPI_COMM_WORLD);
  //printf("Task %d: MPI_Alltoall returned %d\n",procid,status);
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
  
  /*printf("Task %d sendbuf: ",procid);
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
  printf("\n");*/

  //using the counts, put each entry in an array to send
  //calculate displacements and everything else needed for the alltoallv.
  std::cout<<"Rank "<<procid<<" Doing final Alltoallv\n";
  status = MPI_Alltoallv(final_sendbuf, sendcounts, sdispls, MPI_INT, final_recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  
  for(int i = 0; i < recvsize; i+=6){
    uint64_t vtx1 = final_recvbuf[i];
    uint64_t vtx2 = final_recvbuf[i+1];
    uint64_t level1 = final_recvbuf[i+2];
    uint64_t level2 = final_recvbuf[i+3];
    uint64_t proc1 = final_recvbuf[i+4];
    uint64_t proc2 = final_recvbuf[i+5];
    //printf("Task %d received entry: vtx1=%d, vtx2=%d, lvl1=%d, lvl2=%d, proc1=%d, proc2=%d\n",procid, vtx1,vtx2,level1,level2,proc1,proc2);
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
  delete [] sendbuf;
  delete [] recvbuf;
  delete [] sdispls;
  delete [] rdispls;
  delete [] sentcount;
  delete [] final_sendbuf;
  delete [] final_recvbuf;
  delete [] sendcounts;
  delete [] recvcounts; 
  std::cout<<"Rank "<<procid<<" Exiting communication function\n";
}

void art_pt_heuristic(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, 
                      uint64_t* parents, uint64_t* levels, uint64_t* art_pt_flags) {
  std::cout<<"Starting art_pt_heuristic\n";
  //Initialize the queue q->queue with nontree edges.
  std::queue<int> lca_data;
  int* visited_edges = new int[g->m_local];
  for(int i = 0; i < g->m_local; i++) visited_edges[i] = 0;
  init_queue_nontree(g,lca_data,parents,levels,visited_edges);
  //printf("Task %d found %d nontree edges\n",procid, lca_data.size()/6);
  //do LCA traversals incrementally, communicating in batches
    //also need to keep track of the edges used in the traversal, to flag both ends of a bridge.
  std::queue<int> send_queue;
  int all_done = 0;
  std::cout<<"Starting LCA Traversals\n";
  while(!all_done){
    lca_traversal(g,lca_data,send_queue,parents,levels,art_pt_flags,visited_edges);
    //printf("Task %d attempting to send %d entries\n",procid,send_queue.size()/6);
    communicate(g,send_queue,lca_data,visited_edges);
    int local_done = lca_data.empty() && send_queue.empty();
    MPI_Allreduce(&local_done, &all_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
  std::cout<<"LCA Traversals completed\n";
  int bridges = 0;
  //any endpoint of an unvisited edge should be marked as a potential articulation point
  std::cout<<"Marking bridges\n";
  for(int i = 0; i < g->n_local; i++){
    for(int j = g->out_degree_list[i]; j < g->out_degree_list[i+1]; j++){
      if(visited_edges[j] == 0){
        int global_j = 0;
        if(g->out_edges[j] < g->n_local) global_j = g->local_unmap[g->out_edges[j]];
        else global_j = g->ghost_unmap[g->out_edges[j] - g->n_local];
        bridges++;
        //printf("Task %d: edge from %d to %d is a bridge\n",procid, g->local_unmap[i], global_j);
        art_pt_flags[i] = 1;
        art_pt_flags[g->out_edges[j]] = 1;
      }
    }
  }
  std::cout<<"Done marking bridges\n";
  printf("Task %d: found %d bridges\n",procid, bridges);
}

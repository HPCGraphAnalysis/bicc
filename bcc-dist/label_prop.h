#ifndef __label_prop_h__
#define __label_prop_h__

#include<mpi.h>
#include<omp.h>
#include "dist_graph.h"

#include<iostream>
#include<vector>
#include<queue>
#include<set>

#define FIRST 0
#define SECOND 2
#define FIRST_SENDER 1
#define SECOND_SENDER 3
#define BCC_NAME 4
extern int procid, nprocs;
extern bool verbose, debug, verify, output;

enum Grounding_Status {FULL=2, HALF =1, NONE=0};

Grounding_Status get_grounding_status(int* label){
  return (Grounding_Status)((label[FIRST] != -1) + (label[SECOND]!=-1));
}

void communicate(dist_graph_t *g, int** labels, std::queue<int>& reg, std::queue<int>& art,uint64_t* potential_artpts){
  //printf("task %d: Attempting to communicate\n",procid);
  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendcnts[i] = 0;
  }
  //printf("task %d: Accessing g->n_local: %d, g->n_total: %d, and g->ghost_tasks: %x\n",procid, g->n_local,g->n_total,g->ghost_tasks);
  for(int i = g->n_local; i < g->n_total; i++){
    sendcnts[g->ghost_tasks[i-g->n_local]] += 6;
  }
  //printf("task %d: done creating sendcnts\n",procid);
  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) recvcnts[i] = 0;
  int status = MPI_Alltoall(sendcnts,1,MPI_INT,recvcnts,1,MPI_INT,MPI_COMM_WORLD);
  
  int* sdispls = new int[nprocs];
  int* rdispls = new int[nprocs];
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
  }
  
  int sendsize = 0;
  int recvsize = 0;
  int* sentcount = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendsize += sendcnts[i];
    recvsize += recvcnts[i];
    sentcount[i] = 0;
  }
  
  int* sendbuf = new int[sendsize];
  int* recvbuf = new int[recvsize];

  //go through all the ghosted vertices, and send their labels to the owning processor
  for(int i = g->n_local; i < g->n_total; i++){
    int proc_to_send = g->ghost_tasks[i-g->n_local];
    int sendbufidx = sdispls[proc_to_send] + sentcount[proc_to_send];
    sentcount[proc_to_send] += 6;
    sendbuf[sendbufidx++] = g->ghost_unmap[i-g->n_local];
    sendbuf[sendbufidx++] = labels[i][FIRST];
    sendbuf[sendbufidx++] = labels[i][FIRST_SENDER];
    sendbuf[sendbufidx++] = labels[i][SECOND];
    sendbuf[sendbufidx++] = labels[i][SECOND_SENDER];
    sendbuf[sendbufidx++] = labels[i][BCC_NAME];
  }
  status = MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);

  //exchange labels between local label array and the labels present in recvbuf.
  int exchangeidx = 0;
  while(exchangeidx < recvsize){
    int gid = recvbuf[exchangeidx++];
    int lid = get_value(g->map,gid);
    int first = recvbuf[exchangeidx++];
    int first_sender = recvbuf[exchangeidx++];
    int second = recvbuf[exchangeidx++];
    int second_sender = recvbuf[exchangeidx++];
    int bcc_name = recvbuf[exchangeidx++];
    Grounding_Status copy_gs = (Grounding_Status)((first != -1)+(second!=-1));
    Grounding_Status owned_gs = get_grounding_status(labels[lid]);
    //printf("Task %d: sentvtx %d: (%d, %d), (%d, %d), gs: %d\n",procid,gid,first,first_sender,second,second_sender,copy_gs);
    //printf("Task %d: ownedvtx %d: (%d, %d), (%d, %d), gs: %d\n",procid,g->local_unmap[lid],labels[lid][FIRST],labels[lid][FIRST_SENDER],labels[lid][SECOND],labels[lid][SECOND_SENDER],owned_gs);
    if(owned_gs < copy_gs){
      labels[lid][FIRST] = first;
      labels[lid][FIRST_SENDER] = first_sender;
      labels[lid][SECOND] = second;
      labels[lid][SECOND_SENDER] = second_sender;
      labels[lid][BCC_NAME] = bcc_name;
    } else if(owned_gs == copy_gs && owned_gs == HALF){
      if(first != labels[lid][FIRST]){
        labels[lid][SECOND] = first;
        labels[lid][SECOND_SENDER] = first_sender;
        labels[lid][BCC_NAME] = bcc_name;//not sure if this actually changes anything (will be -1?)
      }
    }
    if(get_grounding_status(labels[lid]) != owned_gs){
      if(potential_artpts[lid]) art.push(lid);
      else reg.push(lid);
    }
  }

  //at this point, the owned vertices are all up to date, need to send them back.
  //recvbuf contains ghosts from each processor, ordered by the processor.
  //if we overwrite the labels with the owned version's labels, we can use the recvbuf as the sendbuf, 
  //and the sendbuf as the recvbuf, for ease.
  int updateidx = 0;
  while(updateidx < recvsize){
    int gid = recvbuf[updateidx++];
    int lid = get_value(g->map,gid);
    recvbuf[updateidx++] = labels[lid][FIRST];
    recvbuf[updateidx++] = labels[lid][FIRST_SENDER];
    recvbuf[updateidx++] = labels[lid][SECOND];
    recvbuf[updateidx++] = labels[lid][SECOND_SENDER];
    recvbuf[updateidx++] = labels[lid][BCC_NAME];
  } 
  //send back the updated labels to the relevant processorss
  status = MPI_Alltoallv(recvbuf,recvcnts,rdispls, MPI_INT, sendbuf,sendcnts,sdispls,MPI_INT,MPI_COMM_WORLD);
  
  updateidx = 0;
  while(updateidx < sendsize){
    //update the ghosts on the current processor
    int gid = sendbuf[updateidx++];
    int lid = get_value(g->map,gid);
    /*int first = recvbuf[updateidx++];
    int first_sender = recvbuf[updateidx++];
    int second = recvbuf[updateidx++];
    int second_sender = recvbuf[updateidx++];
    int bcc_name = recvbuf[updateidx++];

    Grounding_Status local_gs = get_grounding_status(labels[lid]);
    Grounding_Status sent_gs = (Grounding_Status)((first != -1)+(second!=-1));
    if(local_gs < sent_gs){
      labels[lid][FIRST] = first;
      labels[lid][FIRST_SENDER] = first_sender;
      labels[lid][SECOND] = second;
      labels[lid][SECOND_SENDER] = second_sender;
      labels[lid][BCC_NAME] = bcc_name;
    } else if (local_gs == sent_gs && sent_gs == HALF){
      if(first != labels[lid][FIRST]){
        labels[lid][SECOND] = second;
        labels[lid][SECOND_SENDER] = second_sender;
        labels[lid][BCC_NAME] = bcc_name;
      }
    }*/
    Grounding_Status local_gs = get_grounding_status(labels[lid]);
    labels[lid][FIRST] = sendbuf[updateidx++];
    labels[lid][FIRST_SENDER] = sendbuf[updateidx++];
    labels[lid][SECOND] = sendbuf[updateidx++];
    labels[lid][SECOND_SENDER] = sendbuf[updateidx++];
    labels[lid][BCC_NAME] = sendbuf[updateidx++];

    if(get_grounding_status(labels[lid]) != local_gs){
      if(potential_artpts[lid])art.push(lid);
      else reg.push(lid);
    }
  }
  delete [] sendcnts;
  delete [] recvcnts;
  delete [] sdispls;
  delete [] rdispls;
  delete [] sentcount;
  delete [] recvbuf;
  delete [] sendbuf;
  //printf("task %d: Concluding communication\n",procid); 
}

//pass labels between two neighboring vertices
void give_labels(dist_graph_t* g,int curr_node, int neighbor, int** labels, bool curr_is_art){
  Grounding_Status curr_gs = get_grounding_status(labels[curr_node]);
  Grounding_Status nbor_gs = get_grounding_status(labels[neighbor]);
  int curr_node_gid = g->local_unmap[curr_node];
  //if the neighbor is full, we don't need to pass labels
  if(nbor_gs == FULL) return;
  //if the current node is empty (shouldn't happen) we can't pass any labels
  if(curr_gs == NONE) return;
  //if the current node is full (and not an articulation point), pass both labels on
  if(curr_gs == FULL && !curr_is_art){
    labels[neighbor][FIRST] = labels[curr_node][FIRST];
    labels[neighbor][FIRST_SENDER] = curr_node_gid;
    labels[neighbor][SECOND] = labels[curr_node][SECOND];
    labels[neighbor][SECOND_SENDER] = curr_node_gid;
    labels[neighbor][BCC_NAME] = labels[curr_node][BCC_NAME];
    return;
  } else if (curr_gs == FULL){
    //if it is an articulation point, and it hasn't sent to this neighbor
    if(labels[neighbor][FIRST_SENDER] != curr_node_gid){
      //send itself as a label
      if(nbor_gs == NONE){
        labels[neighbor][FIRST] = curr_node_gid;
        labels[neighbor][FIRST_SENDER] = curr_node_gid;
      } else if (nbor_gs == HALF){
        if(labels[neighbor][FIRST] != curr_node_gid){
          labels[neighbor][SECOND] = curr_node_gid;
          labels[neighbor][SECOND_SENDER] = curr_node_gid;
        }
      }
    }
    return;
  }
  //if the current node has only one label
  if(curr_gs == HALF){
    //pass that on appropriately
    if(nbor_gs == NONE){
      labels[neighbor][FIRST] = labels[curr_node][FIRST];
      labels[neighbor][FIRST_SENDER] = curr_node_gid;
      labels[neighbor][BCC_NAME] = labels[curr_node][BCC_NAME];
    } else if(nbor_gs == HALF){
      //make sure you aren't giving a duplicate label,
      //and that you haven't sent a label to this neighbor before
      if(labels[neighbor][FIRST] != labels[curr_node][FIRST] && labels[neighbor][FIRST_SENDER] != curr_node_gid){
        labels[neighbor][SECOND] = labels[curr_node][FIRST];
        labels[neighbor][SECOND_SENDER] = curr_node_gid;
        labels[neighbor][BCC_NAME] = labels[curr_node][BCC_NAME];
      }
    }
  }
}

void bfs_prop(dist_graph_t *g, std::queue<int>& reg_frontier,std::queue<int>& art_frontier, int** labels, uint64_t* potential_artpts){
  //printf("task %d: starting propagation sweep\n",procid);
  int done = 0;
  while(!done){
    std::queue<int>* curr = &reg_frontier;
    if(reg_frontier.empty()) curr = &art_frontier;
    while(!curr->empty()){
      int curr_node = curr->front();
      curr->pop();
      //if the current node is a ghost, it should not propagate
      if(curr_node >= g->n_local) continue;

      int out_degree = out_degree(g, curr_node);
      uint64_t* outs = out_vertices(g,curr_node);
     
      //printf("Task %d: starting to check neighbors\n",procid); 
      for(int i = 0; i < out_degree; i++){
        int neighbor = outs[i];
        //printf("Task %d: checking %d to %d\n",procid, curr_node, neighbor);
        Grounding_Status old_gs = get_grounding_status(labels[neighbor]);
        
        //printf("task %d: starting give labels\n",procid); 
        //give curr_node's neighbor some more labels
        give_labels(g,curr_node, neighbor, labels, potential_artpts[curr_node] >= 1);
        //printf("task %d: ending give labels\n",procid);
        
        if(old_gs != get_grounding_status(labels[neighbor])){
          if(potential_artpts[neighbor]) art_frontier.push(neighbor);
          else reg_frontier.push(neighbor);
        }
      }
      
      if(curr->empty()){
        if(curr == &reg_frontier) curr = &art_frontier;
        else curr = &reg_frontier;
      }
      
    }
    //printf("Task %d: BEFORE COMMUNICATE\n",procid);
    for(int i = 0; i < g->n_total; i++){
      int gid = g->local_unmap[i];
      if(i >= g->n_local) gid = g->ghost_unmap[i-g->n_local];
      //printf("Task %d: vtx %d (%d, %d), (%d, %d)\n",procid, gid,labels[i][0],labels[i][1], labels[i][2], labels[i][3]);
    }
    //communicate from owned to ghosts and back
    communicate(g,labels, reg_frontier,art_frontier,potential_artpts);
    //printf("Task %d: AFTER COMMUNICATE\n",procid);
    for(int i = 0; i < g->n_total; i++){
      int gid = g->local_unmap[i];
      if(i >= g->n_local) gid = g->ghost_unmap[i-g->n_local];
      //printf("Task %d: vtx %d (%d, %d), (%d, %d)\n",procid, gid,labels[i][0],labels[i][1], labels[i][2], labels[i][3]);
    }
    
    //do MPI_Allreduce on each processor's local done values
    int local_done = reg_frontier.empty() && art_frontier.empty();
    MPI_Allreduce(&local_done,&done,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
  }
  //printf("task %d: finished propagation sweep\n",procid);
}

int* propagate(dist_graph_t *g, std::queue<int>&reg_frontier,std::queue<int>&art_frontier,int**labels,uint64_t*potential_artpts){
  //propagate initially
  //printf("task %d: initiating propagation\n",procid);
  bfs_prop(g,reg_frontier,art_frontier,labels,potential_artpts);
  //printf("task %d: after initial propagation sweep\n",procid);
  for(int i = 0; i < g->n_total; i++){
    int gid = g->local_unmap[i];
    if(i >= g->n_local) gid = g->ghost_unmap[i-g->n_local];
    //printf("Task %d: vtx %d (%d, %d), (%d, %d)\n",procid, gid,labels[i][0],labels[i][1], labels[i][2], labels[i][3]);
  }
  //fix incomplete propagation
  while(true){
    //printf("task %d: fixing incomplete propagation\n",procid);
    //check for incomplete propagation
    for(int i = 0; i < g->n_local; i++){
      if(potential_artpts[i] && get_grounding_status(labels[i]) == FULL){
        int out_degree = out_degree(g, i);
        uint64_t* outs = out_vertices(g, i);
        for(int j = 0; j < out_degree; j++){
          int neighbor = outs[j];
          if(get_grounding_status(labels[neighbor]) == HALF && labels[neighbor][FIRST] != i && labels[neighbor][FIRST_SENDER] == i){
            reg_frontier.push(i);
          }
        }
      }
    }
    int local_done = reg_frontier.empty();
    int done = 0;
    MPI_Allreduce(&local_done,&done,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    //if(!done) printf("task %d: continuing to fix incomplete propagation\n",procid);
    //else printf("task %d: done fixing incomplete propagation\n",procid);
    //if no incomplete propagation, we're done
    if(done) break;
    //clear out half-full labels
    for(int i = 0; i < g->n_total; i++){
      if(get_grounding_status(labels[i]) == HALF){
        labels[i][FIRST] = -1;
        labels[i][FIRST_SENDER] = -1;
        labels[i][SECOND] = -1;
        labels[i][SECOND_SENDER] = -1;
        labels[i][BCC_NAME] = -1;
      }
    }
    //repropagate
    bfs_prop(g,reg_frontier,art_frontier,labels,potential_artpts);
  }
  
  int* removed = new int[g->n_total];
  for(int i = 0; i < g->n_total; i++){
    Grounding_Status gs = get_grounding_status(labels[i]);
    if(gs == FULL) removed[i] = -2;
    else if(gs == HALF) removed[i] = labels[i][FIRST];
    else removed[i] = -1;
  }
  //printf("task %d: finishing propagation\n",procid);
  return removed;
}

int** bcc_bfs_prop_driver(dist_graph_t *g, uint64_t* potential_artpts, int**labels, int* articulation_point_flags){
  int done = 0;
  std::queue<int> reg_frontier;
  std::queue<int> art_frontier;
  std::queue<int> art_queue;
  int bcc_count = 0;
  //int* articulation_point_flags = new int[g->n_local];
  for(int i = 0; i < g->n_local; i ++){
    articulation_point_flags[i] = 0;
  }
  
  //while there are empty vertices
  while(!done){
    //printf("task %d: initial grounding\n",procid);
    //see how many articulation points all processors know about
    int globalArtPtCount = 0;
    int localArtPtCount = art_queue.size();
    MPI_Allreduce(&localArtPtCount,&globalArtPtCount,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    
    //if none, no one is making progress, so ground two empty neighbors
    if(globalArtPtCount == 0){
      //search for a pair of empty vertices where one is ghosted on a processor
      int foundGhostPair = 0;
      int ownedVtx = -1, ghostVtx = -1;
      for(int i = 0; i < g->n_local; i++){
        if(get_grounding_status(labels[i]) == NONE){
          int out_degree = out_degree(g,i);
          uint64_t* outs = (out_vertices(g,i));
          for(int j = 0; j < out_degree; j++){
            if(outs[j] >= g->n_local){
              //the neighbor is ghosted
              if(get_grounding_status(labels[outs[j]])==NONE){
                ownedVtx = i;
                ghostVtx = outs[j];
                foundGhostPair = 1;
                break;
              }
            }
          }
        }
        if(foundGhostPair) break;
      }
      //if you found a ghost pair, send your own procID, otherwise send -1
      int neighborProc = -1;
      int neighborSend = -1;
      if(foundGhostPair) neighborSend = procid;
      MPI_Allreduce(&neighborSend, &neighborProc,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);

      //if neighborProc is me, I have to ground the neighbors
      if(neighborProc == procid){
        int firstNeighbor_gid = g->local_unmap[ownedVtx];
        int secondNeighbor_gid = g->ghost_unmap[ghostVtx-g->n_local];
        labels[ownedVtx][FIRST] = firstNeighbor_gid;
        labels[ownedVtx][FIRST_SENDER] = firstNeighbor_gid;
        labels[ownedVtx][BCC_NAME] = bcc_count*nprocs + procid;
        labels[ghostVtx][FIRST] = secondNeighbor_gid;
        labels[ghostVtx][FIRST_SENDER] = secondNeighbor_gid;
        labels[ghostVtx][BCC_NAME] = bcc_count*nprocs+procid;
        reg_frontier.push(ownedVtx);
        reg_frontier.push(ghostVtx);
      } else if(neighborProc == -1){
        int foundEmptyPair = 0;
        int vtx1 = -1, vtx2 = -1;
        //if none are found, find any pair of empty vertices
        for(int i = 0; i < g->n_local; i++){
          if(get_grounding_status(labels[i]) == NONE){
            int out_degree = out_degree(g, i);
            uint64_t* outs = (out_vertices(g, i));
            for(int j = 0; j < out_degree; j++){
              if(get_grounding_status(labels[outs[j]]) == NONE){
                foundEmptyPair = 1;
                vtx1 = i;
                vtx2 = outs[j];
                break;
              }
            }
          }
          if(foundEmptyPair) break;
        }
        
        int emptyProc = -1;
        int emptySend = -1;
        if(foundEmptyPair) {
          emptySend = procid;
          printf("task %d: found %d and %d to ground\n",procid,vtx1,vtx2);
        }
        MPI_Allreduce(&emptySend,&emptyProc,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
         
        if(emptyProc == -1){
          break; // we're done here
        } else if(emptyProc == procid){
          int firstNeighbor_gid = g->local_unmap[vtx1];
          int secondNeighbor_gid = g->local_unmap[vtx2];
          labels[vtx1][FIRST] = firstNeighbor_gid;
          labels[vtx1][FIRST_SENDER] = firstNeighbor_gid;
          labels[vtx1][BCC_NAME] = bcc_count*nprocs + procid;
          labels[vtx2][FIRST] = secondNeighbor_gid;
          labels[vtx2][FIRST_SENDER] = secondNeighbor_gid;
          labels[vtx2][BCC_NAME] = bcc_count*nprocs + procid;
          reg_frontier.push(vtx1);
          reg_frontier.push(vtx2);
        }
      }
    } else {
      
      //if this processor knows about articulation points
      if(!art_queue.empty()){
        //look at the front, and ground a neighbor.
        int art_pt = art_queue.front();
        int out_degree = out_degree(g, art_pt);
        uint64_t* outs = (out_vertices(g, art_pt));
        for(int i = 0; i < out_degree; i++){
          if(get_grounding_status(labels[outs[i]]) == NONE){
            int neighbor_gid = -1;
            if(outs[i] >= g->n_local) neighbor_gid = g->ghost_unmap[outs[i]-g->n_local];
            else neighbor_gid = g->local_unmap[outs[i]];
            labels[outs[i]][FIRST] = neighbor_gid;
            labels[outs[i]][FIRST_SENDER] = neighbor_gid;
            labels[outs[i]][BCC_NAME] = bcc_count*nprocs + procid;
            reg_frontier.push(art_pt);
            reg_frontier.push(outs[i]);
            break;
          }
        }
      }
    } 
    //printf("task %d: completed grounding\n",procid);
    //communicate the grounding to other (maybe push this out of the whole grounding?)
    
    communicate(g,labels, reg_frontier,art_frontier,potential_artpts);
    
    //printf("task %d: BEFORE Prop\n",procid);
    //for(int i = 0; i < g->n_total; i++){
    //  printf("task %d: %d, %d; %d, %d; %d\n",procid, labels[i][0],labels[i][1], labels[i][2],labels[i][3],labels[i][4]);
    //}
    int* t = propagate(g,reg_frontier,art_frontier,labels,potential_artpts);
    //printf("task %d: AFTER Prop\n",procid);
    //for(int i = 0; i < g->n_total; i++){
    //  printf("task %d: %d, %d; %d, %d; %d\n",procid, labels[i][0],labels[i][1], labels[i][2],labels[i][3],labels[i][4]);
    //}
    //break;
     
    bcc_count++;
    delete [] t;
    
    //check for owned articulation points
    for(int i = 0; i < g->n_local; i++){
      if(get_grounding_status(labels[i]) == FULL){
        int out_degree = out_degree(g, i);
        uint64_t* outs = (out_vertices(g, i));
        for(int j =0; j < out_degree; j++){
          if(get_grounding_status(labels[outs[j]]) < FULL){
            art_queue.push(i);
            articulation_point_flags[i] = 1;
            break;
          }
        }
      }
    }
    //clear half labels
    for(int i = 0; i < g->n_total; i++){
        if(get_grounding_status(labels[i]) == HALF){
          labels[i][FIRST] = -1;
          labels[i][FIRST_SENDER] = -1;
          labels[i][BCC_NAME] = -1;
        }
    }
    //pop articulation points off of the art_queue if necessary
    if(!art_queue.empty()){
      bool pop_art = true;
      while(pop_art && !art_queue.empty()){
        int top_art_degree = out_degree(g,art_queue.front());
        uint64_t* art_outs = (out_vertices(g,art_queue.front()));
        
        for(int i = 0; i < top_art_degree; i++){
          if(get_grounding_status(labels[art_outs[i]]) < FULL){
            pop_art = false;
          }
        }
        if(pop_art) art_queue.pop();
      }
    }
    
    //check for empty labels
    int emptylabelcount = 0;
    for(int i = 0; i < g->n_total; i++){
      emptylabelcount += (get_grounding_status(labels[i])==NONE);
    }
    int all_empty = 0;
    MPI_Allreduce(&emptylabelcount,&all_empty,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(all_empty == 0) break;
  }
  //std::cout<<me<<": found "<<bcc_count<<" biconnected components\n";
  return labels;
  
}

//int* ice_bfs_prop_driver(graph *g, int *boundary_flags, int *ground_flags){
  
//}

#endif

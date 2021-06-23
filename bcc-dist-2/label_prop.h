#ifndef __label_prop_h__
#define __label_prop_h__

#include<mpi.h>
#include<omp.h>
#include "dist_graph.h"

#include<iostream>
#include<vector>
#include<queue>
#include<set>
#include<algorithm>

extern int procid, nprocs;
extern bool verbose, debug, verify, output;


//do abbreviated LCA traversals to higher-level LCAs
//need the queue because if the traversal is incomplete, we resume it later
//(there is a situation where another traversal would allow progress, but this approach is more general)
bool reduce_labels(dist_graph_t *g, uint64_t curr_vtx, uint64_t* levels, std::vector<std::set<uint64_t>>& LCA_labels,
		   std::unordered_map<uint64_t, std::set<uint64_t>>& remote_LCA_labels,
		   std::unordered_map<uint64_t, uint64_t>& remote_LCA_levels,
		   std::queue<uint64_t>* prop_queue){
   //reduce the lowest-level label until all LCA labels point to the same LCA vertex
   //std::set<uint64_t> curr_vtx_labels = LCA_labels[curr_vtx];
   bool done = false;
   while(!done){
     //if there's only one label, we're completely reduced.
     if(LCA_labels[curr_vtx].size() == 1){
       done = true; //slightly redundant, but whatever.
       break;
     }
     
     uint64_t highest_level_gid = *LCA_labels[curr_vtx].begin();
     uint64_t highest_level = 0;
     //see if the label is remote or local, set levels accordingly
     if(get_value(g->map, highest_level_gid) == NULL_KEY){
       highest_level = remote_LCA_levels[highest_level_gid];
     } else {
       highest_level = levels[ get_value(g->map, highest_level_gid) ];
     }


     for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); ++it){
       uint64_t curr_level = 0;
       //set the level correctly, depending on whether or not this is a remote LCA.
       if(get_value(g->map, highest_level_gid) == NULL_KEY){
         curr_level = remote_LCA_levels[ *it ];
       } else {
         curr_level = levels[ get_value(g->map, *it) ];
       }

       if(curr_level > highest_level){
         highest_level_gid = *it;
	 highest_level = curr_level;
       }
     }
     std::cout<<"\thighest level LCA label is "<<highest_level_gid<<"\n";
     std::cout<<"\t LCA_labels["<<curr_vtx<<".size() = "<<LCA_labels[curr_vtx].size()<<"\n";
     //we aren't done and we need to reduce the highest-level-valued label
     if(LCA_labels[curr_vtx].size() != 1){
       //remove the highest level label, to be replaced by its label.
       LCA_labels[curr_vtx].erase(highest_level_gid);
       //look up the labels of the highest level label ID
       std::set<uint64_t> labels_of_highest_label;
       //if remote, use the remote labels
       if(get_value(g->map, highest_level_gid) == NULL_KEY){
         labels_of_highest_label = remote_LCA_labels[ highest_level_gid ];
       } else { //use the local ones.
         labels_of_highest_label = LCA_labels[ get_value(g->map, highest_level_gid) ];
       }
       std::cout<<"\t labels_of_highest_label.size() = "<<labels_of_highest_label.size()<<"\n";
       //if it has zero or multiple labels, we can't currently use it.
       if(labels_of_highest_label.size() != 1){
	 //save the progress we've made, and try again later
	 LCA_labels[curr_vtx].insert(highest_level_gid);
         prop_queue->push(curr_vtx);
	 //report an incomplete reduction
	 return false;
       } else {
	 //update the label in curr_vtx.
         LCA_labels[curr_vtx].insert(*labels_of_highest_label.begin());
       }
     }
   }
   
   //if we get here, reduction was successful.
   return true;
}

//pass LCA and low labels between two neighboring vertices
void pass_labels(dist_graph_t* g,uint64_t curr_vtx, uint64_t nbor, std::vector<std::set<uint64_t>>& LCA_labels,
		 std::unordered_map<uint64_t, std::set<uint64_t>>& remote_LCA_labels,
		 uint64_t* low_labels, uint64_t* levels, uint64_t* potential_artpts, bool* potential_artpt_did_prop_lower,
		 std::queue<uint64_t>* prop_queue, std::set<uint64_t>& verts_to_send, std::set<uint64_t>& labels_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& procs_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& LCA_procs_to_send){
  //if curr_vert is an potential_artpt
  //  if nbor has a level <= curr_vert
  //    pass received LCA labels to nbor (if different)
  //  else
  //    pass only GID of curr_vert to nbor (if different)
  //  pass low_label to nbor if LCA is the same.
  //  
  //else
  //  pass LCA labels to nbor (if different)
  //  pass low label to nbor if LCA is the same.
  //
  //if nbor was updated, add to prop_queue
  bool nbor_changed = false;
  
  if(potential_artpts[curr_vtx] != 0){
    if(levels[nbor] <= levels[curr_vtx]){
      //see if curr_vtx has any labels that nbor doesn't
      std::vector<uint64_t> diff;
      std::set_difference(LCA_labels[curr_vtx].begin(), LCA_labels[curr_vtx].end(),
		          LCA_labels[nbor].begin(), LCA_labels[nbor].end(),
			  std::inserter(diff, diff.begin()));
      //if so, pass missing IDs to nbor
      if(diff.size() > 0){
        for(int i = 0; i < diff.size(); i++){
          //std::cout<<"LCA vertex "<<curr_vtx<<" giving label "<<diff[i]<<" to vertex "<<nbor<<"\n";
	  //don't give a vertex its own label, it causes headaches in label reduction.
          if(diff[i] != g->local_unmap[nbor]) {
	    LCA_labels[nbor].insert(diff[i]);
	    nbor_changed = true;
	  }
	}
      }
    } else {
      //ONLY propagate to lower level neighbors if it has not 
      //happened up until this point. We can't check the contents
      //of the label, because reductions may eliminate the ID of this
      //LCA from its lower neighbors' labels. re-adding this label later would
      //trigger more reductions than necessary.
      if(!potential_artpt_did_prop_lower[curr_vtx]){
	std::cout<<"LCA vertex "<<curr_vtx<<" passing own ID to lower neighbors\n";
        LCA_labels[nbor].insert(g->local_unmap[curr_vtx]);
	nbor_changed = true;
      }
    }
    //pass low_label to neighbor if LCA_labels are the same, and if it is lower.
    //****MAY WANT TO MAKE SURE LABELS ARE REDUCED BEFORE THIS?******
    if(LCA_labels[curr_vtx] == LCA_labels[nbor] &&
       levels[nbor] <= levels[curr_vtx] &&
       levels[low_labels[curr_vtx]] >  levels[low_labels[nbor]]){
      low_labels[nbor] = low_labels[curr_vtx];
      nbor_changed = true;
    }
  } else {
    std::vector<uint64_t> diff;
    std::set_difference(LCA_labels[curr_vtx].begin(), LCA_labels[curr_vtx].end(),
		        LCA_labels[nbor].begin(), LCA_labels[nbor].end(),
			std::inserter(diff, diff.begin()));
    if(diff.size() > 0){
      for(int i = 0; i < diff.size(); i++){
	//don't give a vertex its own label, it causes headaches in label reduction.
        if(diff[i] != g->local_unmap[nbor]) {
          LCA_labels[nbor].insert(diff[i]);
          nbor_changed = true;
	}
      }
    }

    //****MAY WANT TO MAKE SURE LABELS ARE REDUCED BEFORE THIS?******
    if(LCA_labels[curr_vtx] == LCA_labels[nbor] &&
       levels[low_labels[curr_vtx]] > levels[low_labels[nbor]]
       || (levels[low_labels[curr_vtx]] == levels[low_labels[nbor]] && 
	   g->local_unmap[curr_vtx] > g->local_unmap[nbor])){
      low_labels[nbor] = low_labels[curr_vtx];
      nbor_changed = true;
    }
  }

  if(nbor_changed){
    //if we need to send this vert to remote procs, 
    //add it to verts_to_send, and its labels to 
    //labels_to_send
    if(procs_to_send[nbor].size() > 0){
      verts_to_send.insert(nbor);
      
      //add the labels of nbor to the list of labels we need to send.
      //This list needs expanded, but not right now. We also need to
      //add the procs to which we're sending nbor to the procs to which we
      //need to send its labels.
      for(auto label_it = LCA_labels[nbor].begin(); label_it != LCA_labels[nbor].end(); label_it++){
	//only add ''direct'' labels to send here
	labels_to_send.insert(*label_it);
	//record which procs will need these verts
	LCA_procs_to_send[*label_it].insert(procs_to_send[nbor].begin(), procs_to_send[nbor].end());
      }
    }
    std::cout<<"adding "<<nbor<<" to queue\n";
    prop_queue->push(nbor);
  }
}

void communicate(dist_graph_t* g,
		 std::set<uint64_t>& verts_to_send, std::set<uint64_t>& labels_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& procs_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& LCA_procs_to_send,
		 std::vector<std::set<uint64_t>>& LCA_labels,
		 uint64_t* low_labels,
		 std::unordered_map<uint64_t, std::set<uint64_t>>& remote_LCA_labels,
		 std::unordered_map<uint64_t, uint64_t>& remote_LCA_levels,
		 uint64_t* levels,
		 std::queue<uint64_t>* prop_queue){
  //loop through labels_to_send, add labels-of-labels to the final set to send,
  //also set their LCA_procs_to_send
  std::set<uint64_t> final_labels_to_send;

  for(auto LCA_GID_it = labels_to_send.begin(); LCA_GID_it != labels_to_send.end(); LCA_GID_it++){
    uint64_t curr_LCA_GID = *LCA_GID_it;
    std::set<uint64_t> curr_label;
    bool LCA_is_remote = get_value(g->map, curr_LCA_GID) == NULL_KEY;
    
    if(LCA_is_remote){
      curr_label = remote_LCA_labels[curr_LCA_GID];
    } else {
      curr_label = LCA_labels[get_value(g->map, curr_LCA_GID)];
    }

    //we exclude LCAs with multiple labels because they need reduced on their
    //owning process, remote procs can't do anything with multiple labels.
    //Also, cuts down on send sizes. TODO: does this work?
    while(curr_label.size() == 1 && *curr_label.begin() != curr_LCA_GID){
      //add the LCA label to the verts to send out
      final_labels_to_send.insert(*curr_label.begin());

      //set procs to send based on the label this loop spawned from.
      //important to do this additively, overwriting may cause problems.
      LCA_procs_to_send[curr_LCA_GID].insert(LCA_procs_to_send[*LCA_GID_it].begin(),
		                             LCA_procs_to_send[*LCA_GID_it].end());
      //update the LCA we're looking at
      curr_LCA_GID = *curr_label.begin();
      //is this LCA remote?
      LCA_is_remote = get_value(g->map, curr_LCA_GID) == NULL_KEY;
      
      //if so, look in the right place for the new label.
      if(LCA_is_remote){
        curr_label = remote_LCA_labels[curr_LCA_GID];
      } else {
        curr_label = LCA_labels[get_value(g->map, curr_LCA_GID)];
      }
    }
    final_labels_to_send.insert(*LCA_GID_it);
  }
  
  int* sendcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) sendcnts[i] = 0;

  //loop through verts_to_send and labels_to_send to setup the sendcounts
  for(auto it = verts_to_send.begin(); it != verts_to_send.end(); it++){
    if(LCA_labels[*it].size() != 1) continue; //exclude ghosts that aren't reduced
    for(auto it2 = procs_to_send[*it].begin(); it2 != procs_to_send[*it].end(); it2++){
      sendcnts[*it2]+=3;
    }
  }

  for(auto it = labels_to_send.begin(); it != labels_to_send.end(); it++){
    for(auto it2 = LCA_procs_to_send[*it].begin(); it2 != LCA_procs_to_send[*it].end(); it2++){
      sendcnts[*it2]+=3;
    }
  }

  int* recvcnts = new int[nprocs];
  MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);

  int sendsize = 0;
  int recvsize = 0;
  int* sdispls = new int[nprocs+1];
  int* rdispls = new int[nprocs+1];
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i < nprocs; i++){
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
    sendsize += sendcnts[i-1];
    recvsize += recvcnts[i-1];
  }

  int* sendbuf = new int[sendsize];
  int* recvbuf = new int[recvsize];
  int* sendidx = new int[nprocs];
  for(int i = 0; i < nprocs; i++) sendidx[i] = sdispls[i];

  //add verts_to_send to the sendbuf
  for(auto it = verts_to_send.begin(); it != verts_to_send.end(); it++){
    if(LCA_labels[*it].size() != 1) continue;

    for(auto it2 = procs_to_send[*it].begin(); it2 != procs_to_send[*it].end(); it2++){
      sendbuf[sendidx[*it2]++] = g->local_unmap[*it];
      sendbuf[sendidx[*it2]++] = *LCA_labels[*it].begin();
      sendbuf[sendidx[*it2]++] = low_labels[*it];
    }
  }
  //add labels_to_send to the sendbuf
  for(auto it = labels_to_send.begin(); it != labels_to_send.end(); it++){
    bool LCA_is_remote = get_value(g->map, *it) == NULL_KEY;

    for(auto it2 = LCA_procs_to_send[*it].begin(); it2 != LCA_procs_to_send[*it].end(); it2++){
      sendbuf[sendidx[*it2]++] = *it;
      if(LCA_is_remote){
        sendbuf[sendidx[*it2]++] = *remote_LCA_labels[*it].begin();
	sendbuf[sendidx[*it2]++] = remote_LCA_levels[*it];
      } else {
        sendbuf[sendidx[*it2]++] = *LCA_labels[get_value(g->map, *it)].begin();
	sendbuf[sendidx[*it2]++] = levels[get_value(g->map, *it)];
      }
    }
  }

  //call alltoallv
  MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);

  //on receiving end, process any vertex we can translate to an LID as a ghost,
  //and any vertex we can't as a remote LCA vertex.TODO: differentiate this better,
  //could have a ghost LCA that sends its label and level?
  for(int i = 0; i < recvsize; i+= 3){
    bool is_vtx_remote_LCA = get_value(g->map,recvbuf[i]) == NULL_KEY;
    if(!is_vtx_remote_LCA){
      uint64_t lid = get_value(g->map, recvbuf[i]);
      LCA_labels[lid].clear();
      LCA_labels[lid].insert(recvbuf[i+1]);
      low_labels[lid] = recvbuf[i+2];
      prop_queue->push(lid);
    } else {
      remote_LCA_labels[recvbuf[i]].clear();
      remote_LCA_labels[recvbuf[i]].insert(recvbuf[i+1]);
      remote_LCA_levels[recvbuf[i]] = recvbuf[i+2];
    }
  }
  verts_to_send.clear();
  labels_to_send.clear();
}

void bcc_bfs_prop_driver(dist_graph_t *g,std::vector<uint64_t>& ghost_offsets, std::vector<uint64_t>& ghost_adjs,
	                 uint64_t* potential_artpts, std::vector<std::set<uint64_t>>& LCA_labels, uint64_t* low_labels, 
			 uint64_t* levels, int* articulation_point_flags,
			 std::unordered_map<uint64_t, std::set<int>>& procs_to_send){
  std::set<uint64_t> verts_to_send;
  std::set<uint64_t> labels_to_send;
  std::queue<uint64_t> prop_queue;
  std::queue<uint64_t> n_prop_queue;
  std::unordered_map<uint64_t, std::set<int>> LCA_procs_to_send;
  //aliases for easy queue switching
  std::queue<uint64_t> * curr_prop_queue = &prop_queue;
  std::queue<uint64_t> * next_prop_queue = &n_prop_queue;
  bool* potential_artpt_did_prop_lower = new bool[g->n_total];
  //all vertices flagged in the LCA traversals
  //can initially start propagating.
  for(uint64_t i = 0; i < g->n_local; i++){
    if(potential_artpts[i] != 0) {
      prop_queue.push(i);
      if(procs_to_send[i].size() > 0){
        verts_to_send.insert(i);
	labels_to_send.insert(g->local_unmap[i]);
      }
    }
    potential_artpt_did_prop_lower[i] = false;
    if(levels[i] == 0) {
      LCA_labels[i].insert(g->local_unmap[i]);
      if(procs_to_send[i].size() > 0){
        verts_to_send.insert(i);
	labels_to_send.insert(g->local_unmap[i]);
      }
    }
  }
  
  //used to hold LCA labels of vertices that are not local
  //to this process at all. Used only for reducing LCA labels.
  //indexed by global ID, which is how LCA labels work, so no translation
  //needed, really.
  std::unordered_map<uint64_t, std::set<uint64_t>> remote_LCA_labels;
  std::unordered_map<uint64_t, uint64_t> remote_LCA_levels;

  //every proc needs to enter this loop,
  //as there are collectives inside it.
  int global_done = 1;

  while(global_done > 0){
    //pop a vertex off the queue
    if(curr_prop_queue->size() > 0){
      uint64_t curr_vtx = curr_prop_queue->front();
      curr_prop_queue->pop();
      std::cout<<"looking at vertex "<<curr_vtx<<"\n"; 
      std::cout<<"LCA_label is ";
      for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); ++it){
        std::cout<<*it<<" ";
      }
      std::cout<<"\n";
      //check if the LCA_labels entry needs reduced
      bool full_reduce = true;
      //only call reduction on verts with more than one LCA label
      if(LCA_labels[curr_vtx].size() > 1){
        full_reduce = reduce_labels(g, curr_vtx, levels, LCA_labels, remote_LCA_labels, remote_LCA_levels, next_prop_queue);
      }
      if(full_reduce){
        //pass LCA and low labels to neighbors,
        //add neighbors to queue if they changed.
        int out_degree = 0;
        uint64_t* nbors = nullptr;
	if(curr_vtx < g->n_local){
	  out_degree = out_degree(g, curr_vtx);
	  nbors = out_vertices(g, curr_vtx);
	} else {
	  out_degree = ghost_offsets[curr_vtx+1] - ghost_offsets[curr_vtx];
	  nbors = &ghost_adjs[ghost_offsets[curr_vtx]];
	}
        for(int nbor_idx = 0; nbor_idx < out_degree; nbor_idx++){
          pass_labels(g,curr_vtx, nbors[nbor_idx], LCA_labels, remote_LCA_labels,low_labels, 
                      levels, potential_artpts, potential_artpt_did_prop_lower,
          	      next_prop_queue, verts_to_send, labels_to_send,
		      procs_to_send, LCA_procs_to_send);
        }

        //if this is the first time the potential artpt has passed its labels to neighbors,
        //set potential_artpt_did_prop_lower[artpt] = true.
        if(potential_artpts[curr_vtx] != 0 && !potential_artpt_did_prop_lower[curr_vtx]){
          potential_artpt_did_prop_lower[curr_vtx] = true;
        }
      }
    }
    
    //communicate any changed label, and the label of any LCA that we're sending
    //to remote processes. Have to send LCA labels because of local reductions.
    communicate(g,verts_to_send, labels_to_send, procs_to_send, LCA_procs_to_send, LCA_labels, low_labels, remote_LCA_labels, remote_LCA_levels, levels,next_prop_queue);

    std::swap(curr_prop_queue, next_prop_queue);

    //if all queues are empty, the loop can be broken
    int done = curr_prop_queue->size();
    global_done = 0;
    MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
  }
  
  //set articulation_point_flags for the caller.
  for(uint64_t i = 0; i < g->n_total; i++){
    printf("Local vertex %lu has LCA label %lu and low label %lu\n",i,*LCA_labels[i].begin(),low_labels[i]);

    /**
     * NOTE: I'm not entirely sure this translation of labels to articulation points holds in all cases.
     *       It seems like it works though, need to test more.
     *
     * TODO: A proof of correctness.
    */

    articulation_point_flags[i] = 0;
    if(potential_artpts[i] != 0){
      int out_degree = out_degree(g, i);
      uint64_t* nbors = out_vertices(g, i);
      for(int nbor = 0; nbor < out_degree; nbor++){
        if(LCA_labels[i] != LCA_labels[nbors[nbor]] || low_labels[i] != low_labels[nbors[nbor]]){
	  articulation_point_flags[i] = 1;
	}
      }
    }
  }
}


#endif

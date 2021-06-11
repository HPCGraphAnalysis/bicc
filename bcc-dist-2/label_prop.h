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
		   std::queue<uint64_t>& prop_queue){
   //reduce the lowest-level label until all LCA labels point to the same LCA vertex
   //std::set<uint64_t> curr_vtx_labels = LCA_labels[curr_vtx];
   bool done = false;
   while(!done){
     //if there's only one label, we're completely reduced.
     if(LCA_labels[curr_vtx].size() == 1){
       done = true; //slightly redundant, but whatever.
       break;
     }
     
     uint64_t highest_level_id = *LCA_labels[curr_vtx].begin();
     uint64_t highest_level = levels[highest_level_id];

     for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); ++it){
       if(levels[*it] > highest_level){
         highest_level_id = *it;
	 highest_level = levels[highest_level_id];
       }
     }
     std::cout<<"\thighest level LCA label is "<<highest_level_id<<"\n";
     std::cout<<"\t LCA_labels["<<curr_vtx<<".size() = "<<LCA_labels[curr_vtx].size()<<"\n";
     //we aren't done and we need to reduce the highest-level-valued label
     if(LCA_labels[curr_vtx].size() != 1){
       //remove the highest level label, to be replaced by its label.
       LCA_labels[curr_vtx].erase(highest_level_id);
       //look up the labels of the highest level label ID
       std::set<uint64_t> labels_of_highest_label = LCA_labels[highest_level_id];
       std::cout<<"\t labels_of_highest_label.size() = "<<labels_of_highest_label.size()<<"\n";
       //if it has zero or multiple labels, we can't currently use it.
       if(labels_of_highest_label.size() != 1){
	 //save the progress we've made, and try again later
	 LCA_labels[curr_vtx].insert(highest_level_id);
         prop_queue.push(curr_vtx);
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
		 uint64_t* low_labels, uint64_t* levels, uint64_t* potential_artpts, bool* potential_artpt_did_prop_lower,
		 std::queue<uint64_t>& prop_queue){
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
	//potential_artpt_did_prop_lower[curr_vtx] = true;
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
    std::cout<<"adding "<<nbor<<" to queue\n";
    prop_queue.push(nbor);
  }
}

void communicate(void){
  //left intentionally blank, for now
  //
  //Thinking of doing an async communication pattern.
  //Only test for completed communications if our local
  //propagation requires it. send async messages as we
  //update ghost and LCA labels. Need more communication
  //infrastructure (datastructs to hold who owns what,
  //MPI request objs, etc.) for that to work.
}

void bcc_bfs_prop_driver(dist_graph_t *g, uint64_t* potential_artpts,
		         std::vector<std::set<uint64_t>>& LCA_labels, uint64_t* low_labels, 
			 uint64_t* levels, int* articulation_point_flags){

  std::queue<uint64_t> prop_queue;
  bool* potential_artpt_did_prop_lower = new bool[g->n_total];
  //all vertices flagged in the LCA traversals
  //can initially start propagating.
  for(uint64_t i = 0; i < g->n_total; i++){
    if(potential_artpts[i] != 0) prop_queue.push(i);
    potential_artpt_did_prop_lower[i] = false;
    if(levels[i] == 0) LCA_labels[i].insert(g->local_unmap[i]);
  }
  
  //every proc needs to enter this loop,
  //as there are collectives inside it.
  int global_done = 1;

  while(global_done > 0){
    //pop a vertex off the queue
    uint64_t curr_vtx = prop_queue.front();
    prop_queue.pop();
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
      full_reduce = reduce_labels(g, curr_vtx, levels, LCA_labels, prop_queue);
    }
    if(full_reduce){
      //pass LCA and low labels to neighbors,
      //add neighbors to queue if they changed.
      int out_degree = out_degree(g, curr_vtx);
      uint64_t* nbors = out_vertices(g, curr_vtx);
      for(int nbor_idx = 0; nbor_idx < out_degree; nbor_idx++){
        pass_labels(g,curr_vtx, nbors[nbor_idx], LCA_labels, low_labels, 
	            levels, potential_artpts, potential_artpt_did_prop_lower,
		    prop_queue);
      }

      //if this is the first time the potential artpt has passed its labels to neighbors,
      //set potential_artpt_did_prop_lower[artpt] = true.
      if(potential_artpts[curr_vtx] != 0 && !potential_artpt_did_prop_lower[curr_vtx]){
        potential_artpt_did_prop_lower[curr_vtx] = true;
      }
      //if this vertex neighbors a ghost, send to 
      //relevant processes asynchronously.
      //if it is an LCA, send to all procs.
      communicate(); //<-- will fill this in later.

      //if all queues are empty, the loop can be broken
      int done = prop_queue.size();
      global_done = 0;
      MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
    }
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

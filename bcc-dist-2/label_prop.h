#ifndef __label_prop_h__
#define __label_prop_h__

#include<mpi.h>
#include<omp.h>
#include "dist_graph.h"

#include<iostream>
#include<fstream>
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
		   std::queue<uint64_t>* prop_queue, std::queue<uint64_t>* irreducible_prop_queue,
		   std::set<uint64_t>& irreducible_verts,
		   bool* did_recv_remote_LCA){

   //reduce the lowest-level label until all LCA labels point to the same LCA vertex
   //std::set<uint64_t> curr_vtx_labels = LCA_labels[curr_vtx];
   bool done = false;
   //if we have no data about any label, this reduction is currently irreducible
   /*for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
     uint64_t curr_LCA = *it;
     //if the LCA is local, we have data for it
     if(get_value(g->map, curr_LCA) == NULL_KEY || get_value(g->map, curr_LCA) >= g->n_local){
       //if the LCA is remote, we may not have data for it
       if(remote_LCA_levels.count(curr_LCA) == 0 || remote_LCA_labels[curr_LCA].size() == 0){
         //there is no level entry for the current LCA label, so declare it irreducible, and move on for now
         irreducible_prop_queue->push(curr_vtx);
         return false;
       }
     }
   }*/
   /*for(auto it = remote_LCA_labels.begin(); it != remote_LCA_labels.end(); it++){
     std::cout<<"Task "<<procid<<": remote LCA "<<it->first<<" has labels: ";
     for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++){
       std::cout<<*it2<<" ";
     }
     std::cout<<"\n";
   }*/
   uint64_t curr_GID = curr_vtx;
   if(curr_vtx < g->n_local) curr_GID = g->local_unmap[curr_vtx];
   else curr_GID = g->ghost_unmap[curr_vtx-g->n_local];
  
   /*std::cout<<"Task "<<procid<<": vertex "<<curr_GID<<" has labels: \n\t";
   for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
     std::cout<<*it<<" ";
   }
   std::cout<<"\n";*/
   
    
   if(irreducible_verts.count(curr_GID) == 1){
     return false;
   }
   
   /*bool labels_have_root = false;
   for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
     if(get_value(g->map, *it) == NULL_KEY){
       if(remote_LCA_levels[*it] == 0) labels_have_root = true;
     } else {
       if(levels[get_value(g->map, *it)] == 0) labels_have_root = true;
     }
   }

   if(labels_have_root){
     auto it = LCA_labels[curr_vtx].begin();
     while( it != LCA_labels[curr_vtx].end()){
       if(get_value(g->map, *it) == NULL_KEY){
         if(remote_LCA_levels[*it] != 0) it = LCA_labels[curr_vtx].erase(it);
	 else it++;
       } else{
         if(levels[get_value(g->map, *it)] != 0) it = LCA_labels[curr_vtx].erase(it);
	 else it++;
       }
     }
     std::cout<<"AFTER REMOVING NONROOT: Task "<<procid<<": vertex "<<curr_GID<<" has labels: \n\t";
     for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
       std::cout<<*it<<" ";
     }
     std::cout<<"\n";
   }*/

   while(!done){
     //if there's only one label, we're completely reduced.
     if(LCA_labels[curr_vtx].size() == 1){
       done = true; //slightly redundant, but whatever.
       break;
     }
     

     uint64_t highest_level_gid = *LCA_labels[curr_vtx].begin();
     uint64_t highest_level = 0;
     //see if the label is remote or local, set levels accordingly
     if(get_value(g->map, highest_level_gid) == NULL_KEY /*|| get_value(g->map, highest_level_gid) >= g->n_local*/){
       highest_level = remote_LCA_levels[highest_level_gid];
     } else {
       highest_level = levels[ get_value(g->map, highest_level_gid) ];
     }


     for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); ++it){
       uint64_t curr_level = 0;
       bool is_remote = false;
       //set the level correctly, depending on whether or not this is a remote LCA.
       if(get_value(g->map, *it) == NULL_KEY /*|| get_value(g->map, *it) >= g->n_local*/){
         curr_level = remote_LCA_levels[ *it ];
	 is_remote = true;
	 if(did_recv_remote_LCA[*it] == false){
	   //can't reduce this, it is irreducible
	   irreducible_verts.insert(curr_GID);
	   irreducible_prop_queue->push(curr_vtx);
	   return false;
	 }
       } else {
         curr_level = levels[ get_value(g->map, *it) ];
       }
       
       /*std::cout<<"\t vertex "<<*it<<" has level "<<curr_level;//<<"\n";
       if(irreducible_verts.count(*it) == 1) std::cout<<" and is irreducible";
       if(get_value(g->map, *it) == NULL_KEY) std::cout<<" and is a remote LCA";
       else if(get_value(g->map, *it) >= g->n_local) std::cout<<" and is a ghost";
       std::cout<<"\n";*/
       
       if(curr_level > highest_level){
         highest_level_gid = *it;
	 highest_level = curr_level;
       }
     }
     
     //std::cout<<"\thighest level LCA label is "<<highest_level_gid<<", level "<<highest_level<<"\n";
     //std::cout<<"\tLCA_labels["<<curr_GID<<"].size() = "<<LCA_labels[curr_vtx].size()<<"\n";
     
     //we aren't done and we need to reduce the highest-level-valued label
     //remove the ID under consideration from the current label
     LCA_labels[curr_vtx].erase(highest_level_gid);
     std::set<uint64_t> labels_of_highest_label;
     if(get_value(g->map, highest_level_gid) == NULL_KEY){
       labels_of_highest_label = remote_LCA_labels[ highest_level_gid ];
     } else {
       labels_of_highest_label = LCA_labels[get_value(g->map, highest_level_gid)];
     }
     
     //std::cout<<"\tLCA_labels["<<highest_level_gid<<"].size() = "<<labels_of_highest_label.size()<<"\n";
     

     uint64_t level_of_labels_of_highest_label = 0;
     if(labels_of_highest_label.size() > 0 && get_value(g->map, *labels_of_highest_label.begin()) == NULL_KEY /*|| get_value(g->map, *labels_of_highest_label.begin()) >= g->n_local*/){
       level_of_labels_of_highest_label = remote_LCA_levels[*labels_of_highest_label.begin()];
     } else if( labels_of_highest_label.size() > 0){
       level_of_labels_of_highest_label = levels[get_value(g->map,*labels_of_highest_label.begin())];
     }
     
     if(labels_of_highest_label.count(curr_GID) == 1 && highest_level > levels[curr_vtx]){
       /*std::cout<<"\tremoving cyclic label from current label, new label is {";
       for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
         std::cout<<*it<<" ";
       }
       std::cout<<"\n";*/
       
       continue; 
     }
     
      
     
     
     if(labels_of_highest_label.size() == 1 /*&& highest_level >= level_of_labels_of_highest_label*/){
       //perform reduction successfully (modulo the check to make sure the levels decrease to prevent infinite looping)
       LCA_labels[curr_vtx].insert(*labels_of_highest_label.begin());
       
       /*std::cout<<"\tlabel has been reduced, new label is {";
       for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
         std::cout<<*it<<" ";
       }
       std::cout<<"}\n";*/
       
     } else {
       LCA_labels[curr_vtx].insert(highest_level_gid);
       if(irreducible_verts.count(highest_level_gid) == 1){
         //this reduction is irreducible. Abort and add to irreducibility queue/set
	 //std::cout<<"\tthis label is irreducible, wait until communication to retry\n";
         irreducible_verts.insert(curr_GID);
	 irreducible_prop_queue->push(curr_vtx);
	 return false;
       } else if(labels_of_highest_label.size() != 1 && get_value(g->map, highest_level_gid) >= g->n_local) {
         //the LCA which has no information is a ghost, so it is irreducible until communication. 
	 //std::cout<<"\tthis label is irreducible, wait until communication to retry\n";
	 irreducible_verts.insert(curr_GID);
	 irreducible_prop_queue->push(curr_vtx);
	 return false;
       } else if(labels_of_highest_label.size() == 0){
	 //in distributed memory, impossible to tell wether this is actually going to be reducible without comm, so save for after comm.
	 //std::cout<<"\tthis label is likely irreducible, wait until communication to retry\n";
         irreducible_verts.insert(curr_GID);
	 irreducible_prop_queue->push(curr_vtx);
	 return false;
       } else if(labels_of_highest_label.size() == 1 && highest_level < level_of_labels_of_highest_label){
	 //std::cout<<"\tthis label may be irreducible, wait until communication to retry\n";
         irreducible_verts.insert(curr_GID);
         irreducible_prop_queue->push(curr_vtx);
         return false;     
       } else { //irreducible_set doesn't contain the unreduced/empty LCA label, so it's local. just wait for it.
         //irreducible flag is not set, but there are multiple labels, cannot tell it's irreducible yet,
	 //put it back on the local prop queue and abort current reduction.
	 //std::cout<<"\thighest label is local and unreduced, resubmit to local queue\n";
	 prop_queue->push(curr_vtx);
	 return false;
       }
     }
   }
   
   //if we get here, reduction was successful.
   //std::cout<<"\t***label has been fully reduced!\n";
   return true;
}

//pass LCA and low labels between two neighboring vertices
void pass_labels(dist_graph_t* g,uint64_t curr_vtx, uint64_t nbor, std::vector<std::set<uint64_t>>& LCA_labels,
		 std::unordered_map<uint64_t, std::set<uint64_t>>& remote_LCA_labels,
		 std::unordered_map<uint64_t, uint64_t>& remote_LCA_levels,//used only for low_label levels TODO: make a purpose-made low_label structure, or rename this one
		 uint64_t* low_labels, uint64_t* levels, uint64_t* potential_artpts, bool* potential_artpt_did_prop_lower,
		 std::queue<uint64_t>* prop_queue, std::set<uint64_t>& verts_to_send, std::set<uint64_t>& labels_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& procs_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& LCA_procs_to_send, bool full_reduce, bool reduction_needed,
		 bool* did_recv_remote_LCA,
		 std::queue<uint64_t>* irreducible_queue){
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
  /*if((nbor < g->n_local && g->local_unmap[nbor] == 140529) ){
    uint64_t lid = 0;
    //if( nbor >= g->n_local && g->ghost_unmap[nbor-g->n_local] == 3950) lid = nbor;
    //if(curr_vtx >= g->n_local && g->ghost_unmap[curr_vtx-g->n_local] == 3950) lid = curr_vtx;

    std::cout<<"BEGIN PASS_LABELS: vertex "<<g->local_unmap[nbor]<<" has LCA label {";
    for(auto it = LCA_labels[nbor].begin(); it != LCA_labels[nbor].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}";
    std::cout<<" and low label "<<low_labels[nbor]<<"\n";
  }*/
  bool nbor_changed = false;

  //if the curr_vtx is an LCA
  if(potential_artpts[curr_vtx] != 0){
    //if the neighboring vertex is higher in the tree
    if((levels[nbor] <= levels[curr_vtx] && full_reduce && reduction_needed) || (levels[nbor] == levels[curr_vtx] && full_reduce)){
      //see if curr_vtx has any labels that nbor doesn't
      std::vector<uint64_t> diff;
      std::set_difference(LCA_labels[curr_vtx].begin(), LCA_labels[curr_vtx].end(),
		          LCA_labels[nbor].begin(), LCA_labels[nbor].end(),
			  std::inserter(diff, diff.begin()));
      //if so, pass missing IDs to nbor
      if(diff.size() > 0){
        for(size_t i = 0; i < diff.size(); i++){
          //std::cout<<"LCA vertex "<<curr_vtx<<" giving label "<<diff[i]<<" to vertex "<<nbor<<"\n";
	  //don't give a vertex its own label, it causes headaches in label reduction.
	  uint64_t nbor_gid = nbor;
	  if(nbor < g->n_local) nbor_gid = g->local_unmap[nbor];
	  else nbor_gid = g->ghost_unmap[nbor - g->n_local];
          if(diff[i] != nbor_gid) {
	    LCA_labels[nbor].insert(diff[i]);
	    nbor_changed = true;
	  }
	}
      }
    } else if(levels[nbor] > levels[curr_vtx]){
      //ONLY propagate to lower level neighbors if it has not 
      //happened up until this point. We can't check the contents
      //of the label, because reductions may eliminate the ID of this
      //LCA from its lower neighbors' labels. re-adding this label later would
      //trigger more reductions than necessary.
      if(!potential_artpt_did_prop_lower[curr_vtx] || (full_reduce && reduction_needed)){
	//std::cout<<"LCA vertex "<<curr_vtx<<" passing own ID to lower neighbors\n";
	if(curr_vtx < g->n_local){
	  if(LCA_labels[nbor].count(g->local_unmap[curr_vtx]) == 0){
            LCA_labels[nbor].insert(g->local_unmap[curr_vtx]);
	    nbor_changed = true;
	  }
	} else {
	  if(LCA_labels[nbor].count(g->ghost_unmap[curr_vtx - g->n_local]) == 0){
	    LCA_labels[nbor].insert(g->ghost_unmap[curr_vtx - g->n_local]);
	    nbor_changed = true;
	  }
	}
	//nbor_changed = true;
      }
    }
    //pass low_label to neighbor if LCA_labels are the same, and if it is lower.
    //****MAY WANT TO MAKE SURE LABELS ARE REDUCED BEFORE THIS?******
    uint64_t curr_gid = curr_vtx;
    if(curr_vtx < g->n_local) curr_gid = g->local_unmap[curr_vtx];
    else curr_gid = g->ghost_unmap[curr_vtx-g->n_local];

    /*if(LCA_labels[curr_vtx] == LCA_labels[nbor] &&
       (levels[nbor] <= levels[curr_vtx] || *LCA_labels[nbor].begin() != curr_gid) &&
       (levels[get_value(g->map,low_labels[curr_vtx])] > levels[get_value(g->map, low_labels[nbor])] || 
	(levels[get_value(g->map,low_labels[curr_vtx])] == levels[get_value(g->map,low_labels[nbor])] && 
	low_labels[curr_vtx] > low_labels[nbor]))){*/
    if(LCA_labels[curr_vtx] == LCA_labels[nbor] &&
		    (levels[nbor] <= levels[curr_vtx] || *LCA_labels[nbor].begin() != curr_gid)){
      uint64_t curr_low_label = low_labels[curr_vtx];
      uint64_t nbor_low_label = low_labels[nbor];
      uint64_t curr_low_label_level = 0;
      if(get_value(g->map, curr_low_label) == NULL_KEY){
        curr_low_label_level = remote_LCA_levels[curr_low_label];
      } else {
        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
      }
      uint64_t nbor_low_label_level = 0;
      if(get_value(g->map, nbor_low_label) == NULL_KEY){
        nbor_low_label_level = remote_LCA_levels[nbor_low_label];
      } else {
        nbor_low_label_level = levels[get_value(g->map,nbor_low_label)];
      }
      
      /*if(procid == 0 && curr_gid == 8 && g->local_unmap[nbor] == 4){
        std::cout<<"******************vertex 8 is looking at vertex 4******************\n";
        std::cout<<"\tvertex 8 has low label "<<curr_low_label<<" which is level "<<curr_low_label_level<<"\n";
	std::cout<<"\tvertex 4 has low label "<<nbor_low_label<<" which is level "<<nbor_low_label_level<<"\n";
      }*/
      if(curr_low_label_level > nbor_low_label_level ||
		      (curr_low_label_level == nbor_low_label_level && curr_low_label > nbor_low_label)){
        
        low_labels[nbor] = low_labels[curr_vtx];
        nbor_changed = true;
      }
    }
  } else {
    //for non-LCA verts, only pass labels if
    if((levels[nbor] > levels[curr_vtx] /*&& LCA_labels[nbor].size() == 0*/) || // the neighbor is lower and has no labels, or
       (levels[nbor] <= levels[curr_vtx] && full_reduce /*&& reduction_needed*/)){ //the neighbor is higher and the current label was recently reduced.
      std::vector<uint64_t> diff;
      std::set_difference(LCA_labels[curr_vtx].begin(), LCA_labels[curr_vtx].end(),
          	        LCA_labels[nbor].begin(), LCA_labels[nbor].end(),
          		std::inserter(diff, diff.begin()));
      if(diff.size() > 0){
        for(size_t i = 0; i < diff.size(); i++){
          //don't give a vertex its own label, it causes headaches in label reduction.
          uint64_t nbor_gid = nbor;
          if(nbor < g->n_local) nbor_gid = g->local_unmap[nbor];
          else nbor_gid = g->ghost_unmap[nbor - g->n_local];

          //check that label of diff[i] doesn't contain the gid of nbor
          std::set<uint64_t> labels_of_diff;
	  uint64_t level_of_diff = 0;
	  if(get_value(g->map, diff[i]) == NULL_KEY){
	    //make sure we have the necessary data for looking up labels and levels
	    if(did_recv_remote_LCA[diff[i]]){
	      labels_of_diff = remote_LCA_labels[diff[i]];
	      level_of_diff = remote_LCA_levels[diff[i]];
	    } else { //this needs to also abort the neighbor loop that calls this function
	      //irreducible_queue->push(curr_vtx);
	      //return;
	    }
	  } else {
	    labels_of_diff = LCA_labels[get_value(g->map,diff[i])];
	    level_of_diff = levels[get_value(g->map, diff[i])];
	  }
          if(diff[i] != nbor_gid && labels_of_diff.count(nbor_gid) == 0 && level_of_diff < levels[nbor]) {
            LCA_labels[nbor].insert(diff[i]);
            nbor_changed = true;
          }
        }
      }
    }
    //****MAY WANT TO MAKE SURE LABELS ARE REDUCED BEFORE THIS?******
    
    /*if(LCA_labels[curr_vtx] == LCA_labels[nbor] &&
       (levels[get_value(g->map,low_labels[curr_vtx])] > levels[get_value(g->map,low_labels[nbor])]
       || (levels[get_value(g->map,low_labels[curr_vtx])] == levels[get_value(g->map,low_labels[nbor])] && 
	   low_labels[curr_vtx] > low_labels[nbor]))){*/
    if(LCA_labels[curr_vtx] == LCA_labels[nbor]){
      uint64_t curr_low_label = low_labels[curr_vtx];
      uint64_t nbor_low_label = low_labels[nbor];
      uint64_t curr_low_label_level = 0;
      if(get_value(g->map, curr_low_label) == NULL_KEY){
        curr_low_label_level = remote_LCA_levels[curr_low_label];
      } else {
        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
      }
      uint64_t nbor_low_label_level = 0;
      if(get_value(g->map, nbor_low_label) == NULL_KEY){
        nbor_low_label_level = remote_LCA_levels[nbor_low_label];
      } else {
        nbor_low_label_level = levels[get_value(g->map, nbor_low_label)];
      }
      
      if(curr_low_label_level > nbor_low_label_level || 
          (curr_low_label_level == nbor_low_label_level && curr_low_label > nbor_low_label)){
        low_labels[nbor] = low_labels[curr_vtx];
        nbor_changed = true;
      }
    }
  }

  if(nbor_changed){
    //if we need to send this vert to remote procs, 
    //add it to verts_to_send, and its labels to 
    //labels_to_send
    uint64_t nbor_gid = nbor;
    if(nbor < g->n_local) nbor_gid = g->local_unmap[nbor];
    else nbor_gid = g->ghost_unmap[nbor - g->n_local];
    uint64_t curr_gid = curr_vtx;
    if(curr_vtx < g->n_local) curr_gid = g->local_unmap[curr_vtx];
    else curr_gid = g->ghost_unmap[curr_vtx-g->n_local];

    /*if(procid == 2 && nbor_gid == 560304){
      std::cout<<"Task "<<procid<<": vertex "<<curr_gid<<" changed label of vertex 560304, LCA labels:\n\t";
      for(auto it = LCA_labels[nbor].begin(); it != LCA_labels[nbor].end(); it++){
        std::cout<<*it<<" ";
      }
    }*/
    if(procs_to_send[nbor].size() > 0){
      verts_to_send.insert(nbor);
      
      //add the labels of nbor to the list of labels we need to send.
      //This list needs expanded, but not right now. We also need to
      //add the procs to which we're sending nbor to the procs to which we
      //need to send its labels.
      for(auto label_it = LCA_labels[nbor].begin(); label_it != LCA_labels[nbor].end(); label_it++){
        //std::cout<<"*******************added label "<<*label_it<<" in the pass_labels phase\n";
	
	//only add ''direct'' labels to send here
	labels_to_send.insert(*label_it);
	//record which procs will need these verts
	LCA_procs_to_send[*label_it].insert(procs_to_send[nbor].begin(), procs_to_send[nbor].end());
	//also include procs that have the LCA as a ghost
	LCA_procs_to_send[*label_it].insert(procs_to_send[*label_it].begin(), procs_to_send[*label_it].end());
      }
      //send low labels
      labels_to_send.insert(low_labels[nbor]);
      LCA_procs_to_send[low_labels[nbor]].insert(procs_to_send[nbor].begin(), procs_to_send[nbor].end());
    }
    //std::cout<<"adding "<<nbor<<" to queue\n";
    prop_queue->push(nbor);
  }
  /*if((nbor >= g->n_local && g->ghost_unmap[nbor-g->n_local] == 3950) || (curr_vtx >= g->n_local && g->ghost_unmap[curr_vtx-g->n_local] == 3950) ){
    uint64_t lid = 0;
    if( nbor >= g->n_local && g->ghost_unmap[nbor-g->n_local] == 3950) lid = nbor;
    if(curr_vtx >= g->n_local && g->ghost_unmap[curr_vtx-g->n_local] == 3950) lid = curr_vtx;

    std::cout<<"END PASS_LABELS: vertex "<<g->ghost_unmap[lid-g->n_local]<<" has LCA label {";
    for(auto it = LCA_labels[lid].begin(); it != LCA_labels[lid].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}";
    std::cout<<" and low label "<<low_labels[nbor]<<"\n";
    while(true);
  }*/
  /*if(g->local_unmap[nbor] == 290601){
    std::cout<<"END PASS_LABELS: vertex 290601 has LCA label {";
    for(auto it = LCA_labels[nbor].begin(); it != LCA_labels[nbor].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}";
    std::cout<<" and low label "<<low_labels[nbor]<<"\n";
    std::cout<<"nbor_changed = "<<nbor_changed<<"\n";
    std::cout<<"procs_to_send[nbor].size() = "<<procs_to_send[nbor].size()<<"\n";
  }*/
  /*if(nbor < g->n_local && g->local_unmap[nbor] == 140529){
    std::cout<<"END PASS_LABELS: vertex "<<g->local_unmap[nbor]<<" has LCA label {";
    for(auto it = LCA_labels[nbor].begin(); it != LCA_labels[nbor].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}";
    std::cout<<" and low label "<<low_labels[nbor]<<"\n";
    if(curr_vtx < g->n_local){
      std::cout<<"PASSED FROM: vertex "<<g->local_unmap[curr_vtx]<<", has LCA label {";
    } else {
      std::cout<<"PASSED FROM: vertex "<<g->ghost_unmap[curr_vtx - g->n_local]<<", has LCA label {";
    }
    for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}";
    std::cout<<" and low label "<<low_labels[curr_vtx]<<"\n";
    std::cout<<"nbor_changed = "<<nbor_changed<<"\n";
    std::cout<<"procs_to_send[nbor].size() = "<<procs_to_send[nbor].size()<<"\n";
  }*/
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
		 std::queue<uint64_t>* prop_queue,
		 std::queue<uint64_t>* irreducible_prop_queue,
		 std::set<uint64_t>& irreducible_verts,
		 bool* potential_artpt_did_prop_lower,
		 uint64_t* potential_artpts,
		 std::vector<long unsigned int>& ghost_offsets,
		 std::vector<uint64_t>& ghost_adjs,
		 bool* did_recv_remote_LCA){
  //loop through labels_to_send, add labels-of-labels to the final set to send,
  //also set their LCA_procs_to_send
  /*for(auto it = verts_to_send.begin(); it != verts_to_send.end(); it++){
    if(LCA_labels[*it].count(337898) > 0 && procs_to_send[*it].count(2) > 0){
      std::cout<<"Task "<<procid<<" sending vertex "<<*it<<" with label 337898 to Task 2\n";
    }
  }*/
  /*if(verts_to_send.count(get_value(g->map, 290601)) == 1){
    std::cout<<"sending global vertex 290601 to process "<<*procs_to_send[get_value(g->map, 290601)].begin();
    std::cout<<" it has LCA label {";
    for(auto it = LCA_labels[get_value(g->map, 290601)].begin(); it != LCA_labels[get_value(g->map,290601)].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}";
    std::cout<<" and low label "<<low_labels[get_value(g->map, 290601)]<<"\n";
  }*/
  /*if(get_value(g->map, 140529) < g->n_local){
    std::cout<<"BEGIN COMM: vertex 140529 has LCA label {";
    for(auto it = LCA_labels[get_value(g->map,140529)].begin(); it != LCA_labels[get_value(g->map, 140529)].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}\n";
  }*/
	
  std::set<uint64_t> final_labels_to_send;
  std::set<uint64_t> labels_to_send_later;
  //std::cout<<"Starting to trace through labels_to_send\n";
  for(auto LCA_GID_it = labels_to_send.begin(); LCA_GID_it != labels_to_send.end(); LCA_GID_it++){
    uint64_t curr_LCA_GID = *LCA_GID_it;
    std::set<uint64_t> curr_label;
    bool LCA_is_remote = get_value(g->map, curr_LCA_GID) == NULL_KEY;
    

    if(LCA_is_remote){
      if(did_recv_remote_LCA[curr_LCA_GID]){
        curr_label = remote_LCA_labels[curr_LCA_GID];
      } else {
        continue;
      }
    } else {
      curr_label = LCA_labels[get_value(g->map, curr_LCA_GID)];
    }


    /*if(curr_label.size() == 1){
      if(curr_LCA_GID == 767958 || curr_LCA_GID == 337898){
        if(LCA_is_remote){
          std::cout<<"Task "<<procid<<": sending label "<<curr_LCA_GID<<", with label "<<*curr_label.begin()<< " and level " <<remote_LCA_levels[curr_LCA_GID]<<"\n";
        } else {
          std::cout<<"Task "<<procid<<": sending label "<<curr_LCA_GID<<", with label "<<*curr_label.begin()<< " and level " <<levels[get_value(g->map,curr_LCA_GID)]<<"\n"; 
        }
        std::cout<<"curr_label.size() == "<<curr_label.size()<<" *curr_label.begin() = "<<*curr_label.begin()<<"\n";
      }
      final_labels_to_send.insert(*LCA_GID_it);
    }*/
    std::set<uint64_t> labels_this_traversal;
    /*if(curr_LCA_GID == 406990 || *curr_label.begin() == 406990 || 
       curr_LCA_GID == 210622 || *curr_label.begin() == 210622 ||
       curr_LCA_GID == 90698  || *curr_label.begin() == 90698  ||
       curr_LCA_GID == 262501 || *curr_label.begin() == 262501){
      std::cout<<"Pre-traversal, spawning LCA label is "<<curr_LCA_GID<<", label is {";
      for(auto it = curr_label.begin(); it!= curr_label.end(); it++) std::cout<<" "<<*it;
      std::cout<<"}\n";
    }*/
    //labels_this_traversal.insert(*LCA_GID_it);
    //we exclude LCAs with multiple labels because they need reduced on their
    //owning process, remote procs can't do anything with multiple labels.
    //Also, cuts down on send sizes. TODO: does this work?
    while(curr_label.size() == 1 /*&& *curr_label.begin() != curr_LCA_GID*/){
      //update the LCA we're looking at
      /*if(curr_LCA_GID == 406990 || *curr_label.begin() == 406990){
        std::cout<<"\tcurr_LCA_GID = "<<curr_LCA_GID<<" *curr_label.begin() = "<<*curr_label.begin()<<"\n";
        if(get_value(g->map, curr_LCA_GID) == NULL_KEY) std::cout<<"is a remote LCA\n";
        std::cout<<"\tcurr_label = {";
	for(auto it = curr_label.begin(); it != curr_label.end(); it++) std::cout<<" "<<*it;
	std::cout<<"}\n";
	std::cout<<"\tLCA_procs_to_send["<<*LCA_GID_it<<"] = ";
	for(auto it = LCA_procs_to_send[*LCA_GID_it].begin(); it != LCA_procs_to_send[*LCA_GID_it].end(); it++){
	  std::cout<<*it<<" ";
	}
	std::cout<<"\n";
	std::cout<<"\tLCA_procs_to_send["<<curr_LCA_GID<<"] = ";
	for(auto it = LCA_procs_to_send[curr_LCA_GID].begin(); it != LCA_procs_to_send[curr_LCA_GID].end(); it++){
	  std::cout<<*it<<" ";
	}
	std::cout<<"\n";
	std::cout<<"\tLCA_procs_to_send["<<*curr_label.begin()<<"] = ";
	for(auto it = LCA_procs_to_send[*curr_label.begin()].begin(); it != LCA_procs_to_send[*curr_label.begin()].end(); it++){
	  std::cout<<*it<<" ";
	}
	std::cout<<"\n";
      }*/
      //if LCA is remote and we have no info on it, do not send!
      if(get_value(g->map, curr_LCA_GID) == NULL_KEY && did_recv_remote_LCA[curr_LCA_GID]==false) break;
      //std::cout<<"************added label "<<curr_LCA_GID<<" in communication preprocessing\n";

      std::vector<uint64_t> diff;
      std::set_difference(LCA_procs_to_send[*LCA_GID_it].begin(), LCA_procs_to_send[*LCA_GID_it].end(),
		          LCA_procs_to_send[curr_LCA_GID].begin(), LCA_procs_to_send[curr_LCA_GID].end(),
			  std::inserter(diff, diff.begin()));

      //if this label was already added during this traversal, following the labels further will not yield any new results
      if(labels_this_traversal.count(curr_LCA_GID) > 0) break;
      else labels_this_traversal.insert(curr_LCA_GID);
      //if(final_labels_to_send.count(curr_LCA_GID) != 0 && diff.size() == 0 LCA_procs_to_send[curr_LCA_GID] == LCA_procs_to_send[*LCA_GID_it]) break;
      //add the LCA label to the verts to send out
      //set procs to send based on the label this loop spawned from.
      //important to do this additively, overwriting may cause problems.
      LCA_procs_to_send[curr_LCA_GID].insert(LCA_procs_to_send[*LCA_GID_it].begin(),
		                             LCA_procs_to_send[*LCA_GID_it].end());

      //this LCA is sendable, add it to the finalized send structure
      final_labels_to_send.insert(curr_LCA_GID);
      
      //advance the current vertex to be the label of the old current vertex
      /*if(*curr_label.begin() == 4294967298){
        
	std::cout<<"curr_LCA_GID "<<curr_LCA_GID<<" has label size "<<curr_label.size()<<" curr_label.begin() = "<<*curr_label.begin()<<"\n";
	if(get_value(g->map, curr_LCA_GID) == NULL_KEY) std::cout<<"vertex "<<curr_LCA_GID<<" is a remote LCA\n";
      }*/
      curr_LCA_GID = *curr_label.begin();

      LCA_procs_to_send[curr_LCA_GID].insert(LCA_procs_to_send[*LCA_GID_it].begin(),
		                             LCA_procs_to_send[*LCA_GID_it].end());
       
      //is this LCA remote?
      LCA_is_remote = get_value(g->map, curr_LCA_GID) == NULL_KEY;
      
      //if so, look in the right place for the new label.
      if(LCA_is_remote){
        curr_label = remote_LCA_labels[curr_LCA_GID];
      } else {
        curr_label = LCA_labels[get_value(g->map, curr_LCA_GID)];
      }
      if(curr_label.size() != 1){
	/*if(curr_LCA_GID == 4294967298){
	  std::cout<<"Adding 4294967298 to labels to send later\n";
	  while(true);
	}*/
        labels_to_send_later.insert(curr_LCA_GID);
      }
      /*if(curr_LCA_GID == 406990 || *curr_label.begin() == 406990){
        std::cout<<"\tcurr_LCA_GID = "<<curr_LCA_GID<<" *curr_label.begin() = "<<*curr_label.begin()<<"\n";
        if(get_value(g->map, curr_LCA_GID) == NULL_KEY) std::cout<<"is a remote LCA\n";
        std::cout<<"\tcurr_label = {";
	for(auto it = curr_label.begin(); it != curr_label.end(); it++) std::cout<<" "<<*it;
	std::cout<<"}\n";
	std::cout<<"\tLCA_procs_to_send["<<*LCA_GID_it<<"] = ";
	for(auto it = LCA_procs_to_send[*LCA_GID_it].begin(); it != LCA_procs_to_send[*LCA_GID_it].end(); it++){
	  std::cout<<*it<<" ";
	}
	std::cout<<"\n";
	std::cout<<"\tLCA_procs_to_send["<<curr_LCA_GID<<"] = ";
	for(auto it = LCA_procs_to_send[curr_LCA_GID].begin(); it != LCA_procs_to_send[curr_LCA_GID].end(); it++){
	  std::cout<<*it<<" ";
	}
	std::cout<<"\n";
	std::cout<<"\tLCA_procs_to_send["<<*curr_label.begin()<<"] = ";
	for(auto it = LCA_procs_to_send[*curr_label.begin()].begin(); it != LCA_procs_to_send[*curr_label.begin()].end(); it++){
	  std::cout<<*it<<" ";
	}
	std::cout<<"\n";
      }*/
    }
    /*if(*LCA_GID_it == 406990 || *LCA_GID_it == 210622 || *LCA_GID_it == 90698 || *LCA_GID_it == 262501){
      std::cout<<"Post-traversal, spawning LCA label was "<<curr_LCA_GID<<", label was {";
      for(auto it = curr_label.begin(); it!= curr_label.end(); it++) std::cout<<" "<<*it;
      std::cout<<"}\n";
    }*/
    
  }

  for(auto it = labels_to_send.begin(); it != labels_to_send.end(); it++){
    if(final_labels_to_send.count(*it) == 0){
      //if(*it == 406990) std::cout<<"Sending vertex "<<*it<<" later\n";
      labels_to_send_later.insert(*it);
    }
  }

  /*if(final_labels_to_send.count(0) > 0){
    std::cout<<"Task "<<procid<<": sending label 0, with label "<< "to tasks:\n";
    for(auto it = LCA_procs_to_send[0].begin(); it != LCA_procs_to_send[0].end(); it++){
      std::cout<<*it<<" ";
    }
    std::cout<<"\n";
  }*/

  for(int p = 0; p < nprocs; p++){
    if(p == procid){
      /*std::cout<<"Task "<<procid<<": final labels to send contains:\n\t";
      for(auto it = final_labels_to_send.begin(); it != final_labels_to_send.end(); it++){
        std::cout<<*it<<" ";
      }
      std::cout<<"\n";*/

      for(auto it = final_labels_to_send.begin(); it != final_labels_to_send.end(); it++){
	if(*it == 406990){
	  /*std::cout<<"Task "<<procid<<": sending LCA "<<*it<<" to processes:\n\t";
          for(auto it2 = LCA_procs_to_send[*it].begin(); it2 != LCA_procs_to_send[*it].end(); it2++){
            std::cout<<*it2<<" ";
	  }
	  std::cout<<"\n";*/
	  /*std::cout<<"vertex "<<*it<<" has "<<LCA_labels[*it].size()<<" labels:\n\t";
	  for(auto it2 = LCA_labels[*it].begin(); it2 != LCA_labels[*it].end(); it2++){
	    std::cout<<*it2<<" ";
	  }
	  std::cout<<"\n";*/
	}
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int* sendcnts = new int[nprocs];
  int* vertrecvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++) {
    sendcnts[i] = 0;
    vertrecvcnts[i] = 0;
  }

  //loop through verts_to_send and labels_to_send to setup the sendcounts
  for(auto it = verts_to_send.begin(); it != verts_to_send.end(); it++){
    if(LCA_labels[*it].size() != 1) continue; //exclude ghosts that aren't reduced
    for(auto it2 = procs_to_send[*it].begin(); it2 != procs_to_send[*it].end(); it2++){
      sendcnts[*it2]+=3;
    }
  }

  MPI_Alltoall(sendcnts, 1, MPI_INT, vertrecvcnts, 1, MPI_INT, MPI_COMM_WORLD);

  for(auto it = final_labels_to_send.begin(); it != final_labels_to_send.end(); it++){
    for(auto it2 = LCA_procs_to_send[*it].begin(); it2 != LCA_procs_to_send[*it].end(); it2++){
      //std::cout<<"Task "<<procid<<": sending vertex "<<*it<<" to process "<<*it2<<"\n";
      sendcnts[*it2]+=3;
    }
  }
  
  /*for(int p = 0; p < nprocs; p++){
    if(p == procid){
      std::cout<<"Task "<<procid<<": sendcnts:\n\t";
      for(int i = 0; i < nprocs; i++){
        std::cout<<sendcnts[i]<<" ";
      }
      std::cout<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/

  int* recvcnts = new int[nprocs];
  MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);

  int sendsize = 0;
  int recvsize = 0;
  int* sdispls = new int[nprocs+1];
  int* rdispls = new int[nprocs+1];
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i = 1; i <= nprocs; i++){
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
  for(auto it = final_labels_to_send.begin(); it != final_labels_to_send.end(); it++){
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
  
  /*for(int p = 0; p < nprocs; p++){
    if(p == procid){
      std::cout<<"Task "<<procid<<": sendbuf:\n\t";
      for(int i = 0; i < sendsize; i++){
        std::cout<<sendbuf[i]<<" ";
      }
      std::cout<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/

  //call alltoallv
  MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);
  
  /*for(int p = 0; p < nprocs; p++){
    if(p == procid){
      std::cout<<"Task "<<procid<<": recvcnts:\n\t";
      for(int i = 0; i < nprocs; i++){
        std::cout<<recvcnts[i]<<" ";
      }
      std::cout<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/

  //on receiving end, process any vertex we can translate to an LID as a ghost,
  //and any vertex we can't as a remote LCA vertex.
  for(int p = 0; p < nprocs; p++){
    for(int ridx = rdispls[p]; ridx < rdispls[p+1]; ridx+=3){
      //remote LCA
      if(ridx - rdispls[p] >= vertrecvcnts[p]){
	/*if(recvbuf[ridx] == 3950 && procid == 1){ 
	  std::cout<<"**** Received LCA vertex "<<recvbuf[ridx]<<" with label "<<recvbuf[ridx+1]<<" and level "<<recvbuf[ridx+2]<<" from process "<<p<<"\n";
	}*/
        did_recv_remote_LCA[recvbuf[ridx]] = true;
	remote_LCA_labels[recvbuf[ridx]].clear();
	remote_LCA_labels[recvbuf[ridx]].insert(recvbuf[ridx+1]);
	remote_LCA_levels[recvbuf[ridx]] = recvbuf[ridx+2]; 
      } else { //ghost
	/*if(recvbuf[ridx]==3950 && procid ==1){
	  std::cout<<"**** Received ghost vertex "<<recvbuf[ridx]<<" with label "<<recvbuf[ridx+1]<<" and low_label "<<recvbuf[ridx+2]<<" from process  "<<p<<"\n";
	}*/
	//if(recvbuf[ridx] == 290601) std::cout<<" Received LCA vertex "<<recvbuf[ridx]<<" with label "<<recvbuf[ridx+1]<<" and level "<<recvbuf[ridx+2]<<"\n";
        uint64_t lid = get_value(g->map, recvbuf[ridx]);
	//if the ghost is an LCA, and the label has updated, reset the propagate-to-lower flag to false, to trigger re-propagation.
	if(/*potential_artpts[lid] != 0 &&*/ *LCA_labels[lid].begin() != recvbuf[ridx+1]){
          potential_artpt_did_prop_lower[lid] = false;
	  // propagate from ghost LCAs, if the label has changed, treat it as a newly reduced label.
          // need: ghost_nbors
	}
	LCA_labels[lid].clear();
	LCA_labels[lid].insert(recvbuf[ridx+1]);
	low_labels[lid] = recvbuf[ridx+2];
	prop_queue->push(lid);
      }
    }
  }
  
  verts_to_send.clear();
  labels_to_send.clear();
  labels_to_send.insert(labels_to_send_later.begin(), labels_to_send_later.end());
  /*for(auto it = final_labels_to_send.begin(); it != final_labels_to_send.end(); it++){
    if(labels_to_send.count(*it) == 0) LCA_procs_to_send[*it].clear();
  }*/
  for(int p = 0; p < nprocs; p++){
    for(int ridx = rdispls[p]; ridx < rdispls[p+1]; ridx+=3){
      if(ridx - rdispls[p] < vertrecvcnts[p]){
        uint64_t lid = get_value(g->map, recvbuf[ridx]);
        if(potential_artpts[lid] != 0 && potential_artpt_did_prop_lower[lid] == false){
          int out_degree = ghost_offsets[lid+1 - g->n_local] - ghost_offsets[lid - g->n_local];
          uint64_t* nbors = &ghost_adjs[ghost_offsets[lid-g->n_local]];
	  for(int i = 0; i < out_degree; i++){
            //pull low labels from neighbors, don't add to send structs, because we don't send ghosts
            /*if(LCA_labels[lid] == LCA_labels[nbors[i]]){
	      uint64_t curr_low_label = low_labels[lid];
	      uint64_t nbor_low_label = low_labels[nbors[i]];
	      uint64_t curr_low_label_level = 0;
	      if(get_value(g->map, curr_low_label) == NULL_KEY){
	        curr_low_label_level = remote_LCA_levels[curr_low_label];
	      } else {
	        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
	      }
	      uint64_t nbor_low_label_level = 0;
	      if(get_value(g->map, nbor_low_label) == NULL_KEY){
	        nbor_low_label_level = remote_LCA_levels[nbor_low_label];
	      } else {
	        nbor_low_label_level = levels[get_value(g->map, nbor_low_label)];
	      }
	      if(curr_low_label_level < nbor_low_label_level ||
		 (curr_low_label_level == nbor_low_label_level && curr_low_label < nbor_low_label)){
	        low_labels[lid] = low_labels[nbors[i]];
	      }

	    }*/
	    //potentially re-propagate from ghost LCAs
            pass_labels(g,lid, nbors[i], LCA_labels, remote_LCA_labels,remote_LCA_levels,
	 	        low_labels, levels, potential_artpts, potential_artpt_did_prop_lower,
          	        prop_queue, verts_to_send, labels_to_send,
		        procs_to_send, LCA_procs_to_send, true, true,did_recv_remote_LCA, irreducible_prop_queue);
	    
	  } 
	} else if (potential_artpt_did_prop_lower[lid] == false){
          uint64_t gid = g->ghost_unmap[lid - g->n_local];
	  /*if(gid == 373238 && low_labels[lid] != 812576){
            std::cout<<"pulling low label to ghost 373238, ";
	    std::cout<<"LCA_label = {";
	    for(auto it = LCA_labels[lid].begin(); it != LCA_labels[lid].end(); it++){
	      std::cout<<" "<<*it;
	    }
	    std::cout<<"}, and low_label "<<low_labels[lid]<<"\n";
	  }*/
	  int out_degree = ghost_offsets[lid+1 - g->n_local] - ghost_offsets[lid - g->n_local];
          uint64_t* nbors = &ghost_adjs[ghost_offsets[lid-g->n_local]];
	  for(int i = 0; i < out_degree; i++){
            uint64_t nbor_gid = nbors[i];
	    if(nbors[i] < g->n_local){
	      nbor_gid = g->local_unmap[nbors[i]];
	    } else {
	      nbor_gid = g->ghost_unmap[nbors[i]-g->n_local];
	    }
            if(LCA_labels[lid] == LCA_labels[nbors[i]] && ((levels[lid] <= levels[nbors[i]]) || 
				                           (potential_artpts[nbors[i]] == 0) ||
							   (potential_artpts[nbors[i]] != 0 && *LCA_labels[lid].begin() != nbor_gid))){
	      uint64_t curr_low_label = low_labels[lid];
	      uint64_t nbor_low_label = low_labels[nbors[i]];
	      uint64_t curr_low_label_level = 0;
	      if(get_value(g->map, curr_low_label) == NULL_KEY){
	        curr_low_label_level = remote_LCA_levels[curr_low_label];
	      } else {
	        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
	      }
	      uint64_t nbor_low_label_level = 0;
	      if(get_value(g->map, nbor_low_label) == NULL_KEY){
	        nbor_low_label_level = remote_LCA_levels[nbor_low_label];
	      } else {
	        nbor_low_label_level = levels[get_value(g->map, nbor_low_label)];
	      }
	      if(curr_low_label_level < nbor_low_label_level ||
		 (curr_low_label_level == nbor_low_label_level && curr_low_label < nbor_low_label)){
	        low_labels[lid] = low_labels[nbors[i]];
	      }

	    }
	  }
	  /*if(gid == 373238){
            std::cout<<"pulling low label to ghost 373238, ";
	    std::cout<<"LCA_label = {";
	    for(auto it = LCA_labels[lid].begin(); it != LCA_labels[lid].end(); it++){
	      std::cout<<" "<<*it;
	    }
	    std::cout<<"}, and low_label "<<low_labels[lid]<<"\n";
	  }*/
	}
	potential_artpt_did_prop_lower[lid] = true;
      }
    }
  }

  /*if(get_value(g->map, 140529) < g->n_local){
    std::cout<<"END COMM: vertex 140529 has LCA label {";
    for(auto it = LCA_labels[get_value(g->map,140529)].begin(); it != LCA_labels[get_value(g->map, 140529)].end(); it++){
      std::cout<<" "<<*it;
    }
    std::cout<<"}\n";
  }*/
  /*std::cout<<"labels_to_send_later contains:\n\t";
  for(auto it = labels_to_send_later.begin(); it != labels_to_send_later.end(); it++){
    std::cout<<*it<<" ";
  }
  std::cout<<"\n";*/
}

void print_labels(dist_graph_t *g, uint64_t vertex, std::vector<std::set<uint64_t>> LCA_labels, uint64_t* low_labels, uint64_t* potential_artpts, uint64_t* levels){

  if(vertex < g->n_local) std::cout<<"vertex "<<g->local_unmap[vertex]<<" has LCA label "<<*LCA_labels[vertex].begin()<<", low label "<<low_labels[vertex]<<" and level "<<levels[vertex];
  else std::cout<<"vertex "<<g->ghost_unmap[vertex - g->n_local]<<" has LCA label "<<*LCA_labels[vertex].begin()<<", low label "<<low_labels[vertex]<<" and level "<<levels[vertex];
  if(vertex >= g->n_local) std::cout<<" and is a ghost";
  if(potential_artpts[vertex] != 0){
    std::cout<<" and is an LCA vertex, which neighbors:\n\t";
  } else std::cout<<" neighbors:\n\t";

  uint64_t vertex_out_degree = out_degree(g, vertex);
  uint64_t* vertex_nbors = out_vertices(g, vertex);
  for(uint64_t i = 0; i < vertex_out_degree; i++){
    if(vertex_nbors[i] < g->n_local) std::cout<<"vertex "<<g->local_unmap[vertex_nbors[i]]<<" has LCA label "<<*LCA_labels[vertex_nbors[i]].begin()<<", low label "
	                                      <<low_labels[vertex_nbors[i]]<<" and level "<<levels[vertex_nbors[i]];
    else std::cout<<"vertex "<<g->ghost_unmap[vertex_nbors[i]-g->n_local]<<" has LCA label "<<*LCA_labels[vertex_nbors[i]].begin()<<", low label "<<low_labels[vertex_nbors[i]]<<" and level "<<levels[vertex_nbors[i]];
    if(potential_artpts[vertex_nbors[i]] != 0){
      std::cout<<" and is an LCA vertex";
    }
    if(vertex_nbors[i] >= g->n_local) std::cout<<" and is a ghost";
    std::cout<<"\n\t";
  }
  std::cout<<"\n";
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
  std::queue<uint64_t> * irreducible_prop_queue = &n_prop_queue;
  bool* potential_artpt_did_prop_lower = new bool[g->n_total];
  bool* did_recv_remote_LCA = new bool[g->n];
  for(uint64_t i = 0; i < g->n; i++) did_recv_remote_LCA[i] = false;
  std::set<uint64_t> irreducible_verts;
  /*if(procid == 1){
    std::cout<<"vertex 3950 has LCA label size " <<LCA_labels[get_value(g->map,3950)].size()<<"\n";
  }*/
  //all vertices flagged in the LCA traversals
  //can initially start propagating.
  //std::cout<<"Task "<<procid<<": starting propagations\n";
  
  for(int i = 0; i < g->n_local; i++){
    if(g->local_unmap[i] == 300879 || g->local_unmap[i] == 607501 || g->local_unmap[i] == 300895
		    || g->local_unmap[i] == 300876){
      std::cout<<"Task "<<procid<<" vertex "<<g->local_unmap[i]<<" level "<<levels[i]<< " neighbors:\n\t";
      int out_degree = out_degree(g,i);
      uint64_t* nbors = out_vertices(g,i);
      for(int j = 0; j < out_degree; j++){
        if(nbors[j] < g->n_local) std::cout<<g->local_unmap[nbors[j]]<<" ("<<levels[nbors[j]];
	else std::cout<<g->ghost_unmap[nbors[j]-g->n_local]<<" ("<<levels[nbors[j]];
	if(potential_artpts[nbors[j]]) std::cout<<"/LCA";
	if(nbors[j] >= g->n_local) std::cout<<"/ghost";
	std::cout<<") ";
      }
      std::cout<<"\n";
    }
  }

  for(uint64_t i = 0; i < g->n_total; i++){
    if(potential_artpts[i] != 0) {
      prop_queue.push(i);
      if(procs_to_send[i].size() > 0){
        verts_to_send.insert(i);
	//std::cout<<"***********adding vertex "<<g->local_unmap[i]<<" to labels_to_send, it is a potential artpt\n";
	labels_to_send.insert(g->local_unmap[i]);
	LCA_procs_to_send[g->local_unmap[i]].insert(procs_to_send[i].begin(), procs_to_send[i].end());
      }
    }
    potential_artpt_did_prop_lower[i] = false;
    if(levels[i] == 0) {
      if(i < g->n_local) LCA_labels[i].insert(g->local_unmap[i]);
      else LCA_labels[i].insert(g->ghost_unmap[i-g->n_local]);
      if(procs_to_send[i].size() > 0){
        verts_to_send.insert(i);
	//std::cout<<"**********adding vertex "<<g->local_unmap[i]<<" to labels_to_send, it is the root\n";
	labels_to_send.insert(g->local_unmap[i]);
	LCA_procs_to_send[g->local_unmap[i]].insert(procs_to_send[i].begin(), procs_to_send[i].end());
      }
    }
    //irreducible_flags[i] = false;
  }
  
  //used to hold LCA labels of vertices that are not local
  //to this process at all. Used only for reducing LCA labels.
  //indexed by global ID, which is how LCA labels work, so no translation
  //needed, really.
  std::unordered_map<uint64_t, std::set<uint64_t>> remote_LCA_labels;
  std::unordered_map<uint64_t, uint64_t> remote_LCA_levels;
  
  /*if(procid == 1){
    std::cout<<"vertex 3950 has LCA label size " <<LCA_labels[get_value(g->map,3950)].size()<<"\n";
  }*/

  //every proc needs to enter this loop,
  //as there are collectives inside it.
  int global_done = 1;

  while(global_done > 0){
    //pop a vertex off the queue
    while(curr_prop_queue->size() > 0){
      uint64_t curr_vtx = curr_prop_queue->front();
      curr_prop_queue->pop();
      uint64_t curr_gid = curr_vtx;
      if(curr_vtx < g->n_local) curr_gid = g->local_unmap[curr_vtx];
      else curr_gid = g->ghost_unmap[curr_vtx - g->n_local];
      /*if( curr_vtx < g->n_local && (curr_gid == 300876 || curr_gid == 300879)){
	std::cout<<"Task "<<procid<<" owns ";
        std::cout<<"vertex "<<curr_gid<<" has labels {";
	for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
		std::cout<<" "<<*it;
	}
	std::cout<<"}\n";
      }*/
      //check if the LCA_labels entry needs reduced
      bool reduction_needed = (LCA_labels[curr_vtx].size() > 1);
      bool full_reduce = true;
      //only call reduction on verts with more than one LCA label
      if(reduction_needed){
	if(curr_gid == 140529){
          std::cout<<"reducing label of vertex 140529, ";
	  std::cout<<"LCA_label = {";
	  for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
	    std::cout<<" "<<*it;
	  }
	  std::cout<<"}, and low_label "<<low_labels[curr_vtx]<<"\n";
	}
        full_reduce = reduce_labels(g, curr_vtx, levels, LCA_labels, remote_LCA_labels, remote_LCA_levels, curr_prop_queue, irreducible_prop_queue, irreducible_verts, did_recv_remote_LCA);
	if(curr_gid == 140529){
          std::cout<<"Finished reducing label of vertex 140529, ";
	  std::cout<<"LCA_label = {";
	  for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
	    std::cout<<" "<<*it;
	  }
	  std::cout<<"}, and low_label "<<low_labels[curr_vtx]<<"\n"; 
	}
	if(full_reduce && procs_to_send[curr_vtx].size() > 0){
	  verts_to_send.insert(curr_vtx);
	  for(auto label_it = LCA_labels[curr_vtx].begin(); label_it != LCA_labels[curr_vtx].end(); label_it++){
	    labels_to_send.insert(*label_it);
	    LCA_procs_to_send[*label_it].insert(procs_to_send[curr_vtx].begin(), procs_to_send[curr_vtx].end());
	    LCA_procs_to_send[*label_it].insert(procs_to_send[*label_it].begin(), procs_to_send[*label_it].end());
	  }
	}
      }

      int out_degree = 0;
      uint64_t* nbors = nullptr;
      if(curr_vtx < g->n_local){
        out_degree = out_degree(g, curr_vtx);
        nbors = out_vertices(g, curr_vtx);
      } else {
        out_degree = ghost_offsets[curr_vtx+1 - g->n_local] - ghost_offsets[curr_vtx - g->n_local];
        nbors = &ghost_adjs[ghost_offsets[curr_vtx-g->n_local]];
      }
        //pass LCA and low labels to neighbors,
        //add neighbors to queue if they changed.
        
	//pull low labels from neighbors if a reduction happened (low label may be out of date)
	if(full_reduce && reduction_needed){
	  if(curr_gid == 140529){
            std::cout<<"pulling low label to 140529, ";
	    std::cout<<"LCA_label = {";
	    for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
	      std::cout<<" "<<*it;
	    }
	    std::cout<<"}, and low_label "<<low_labels[curr_vtx]<<"\n";
	  }
	  bool curr_changed = false;
	  for(int nbor_idx = 0; nbor_idx < out_degree; nbor_idx++){
	    uint64_t nbor = nbors[nbor_idx];
            
            uint64_t nbor_gid = nbors[nbor_idx];
	    if(nbors[nbor_idx] < g->n_local){
	      nbor_gid = g->local_unmap[nbors[nbor_idx]];
	    } else {
	      nbor_gid = g->ghost_unmap[nbors[nbor_idx]-g->n_local];
	    }
            if(LCA_labels[curr_vtx] == LCA_labels[nbors[nbor_idx]] && ((levels[curr_vtx] <= levels[nbors[nbor_idx]]) || 
				                                      (potential_artpts[nbors[nbor_idx]] == 0) ||
							              (potential_artpts[nbors[nbor_idx]] != 0 && *LCA_labels[curr_vtx].begin() != nbor_gid))){
              uint64_t curr_low_label = low_labels[curr_vtx];
              uint64_t nbor_low_label = low_labels[nbor];
              uint64_t curr_low_label_level = 0;
              if(get_value(g->map, curr_low_label) == NULL_KEY){
                curr_low_label_level = remote_LCA_levels[curr_low_label];
              } else {
                curr_low_label_level = levels[get_value(g->map, curr_low_label)];
              }
              uint64_t nbor_low_label_level = 0;
              if(get_value(g->map, nbor_low_label) == NULL_KEY){
                nbor_low_label_level = remote_LCA_levels[nbor_low_label];
              } else {
                nbor_low_label_level = levels[get_value(g->map, nbor_low_label)];
              }
              
              if(curr_low_label_level < nbor_low_label_level || 
                  (curr_low_label_level == nbor_low_label_level && curr_low_label < nbor_low_label)){
                low_labels[curr_vtx] = low_labels[nbor];
                curr_changed = true;
              }
            }
	  }
	  if(curr_gid == 140529){
            std::cout<<"pulling low label to 140529, ";
	    std::cout<<"LCA_label = {";
	    for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
	      std::cout<<" "<<*it;
	    }
	    std::cout<<"}, and low_label "<<low_labels[curr_vtx]<<"\n";
	  }
	  if(curr_changed && procs_to_send[curr_vtx].size() > 0){
	    //send curr_vtx & labels to remotes if reduction happened and low label updated
	    verts_to_send.insert(curr_vtx);
	    for(auto label_it = LCA_labels[curr_vtx].begin(); label_it != LCA_labels[curr_vtx].end(); label_it++){
	      labels_to_send.insert(*label_it);
	      LCA_procs_to_send[*label_it].insert(procs_to_send[curr_vtx].begin(), procs_to_send[curr_vtx].end());
	      LCA_procs_to_send[*label_it].insert(procs_to_send[*label_it].begin(), procs_to_send[*label_it].end());
	    }
            labels_to_send.insert(low_labels[curr_vtx]);
            LCA_procs_to_send[low_labels[curr_vtx]].insert(procs_to_send[curr_vtx].begin(), procs_to_send[curr_vtx].end());
	  }
	}

        for(int nbor_idx = 0; nbor_idx < out_degree; nbor_idx++){
          pass_labels(g,curr_vtx, nbors[nbor_idx], LCA_labels, remote_LCA_labels,remote_LCA_levels,
		      low_labels, levels, potential_artpts, potential_artpt_did_prop_lower,
          	      curr_prop_queue, verts_to_send, labels_to_send,
		      procs_to_send, LCA_procs_to_send, full_reduce, reduction_needed, did_recv_remote_LCA, irreducible_prop_queue);
        }

        //if this is the first time the potential artpt has passed its labels to neighbors,
        //set potential_artpt_did_prop_lower[artpt] = true.
        if(potential_artpts[curr_vtx] != 0 && !potential_artpt_did_prop_lower[curr_vtx]){
          potential_artpt_did_prop_lower[curr_vtx] = true;
        }
      //}
    }
    //std::cout<<"Task "<<procid<<": communicating...\n"; 
    
    //communicate any changed label, and the label of any LCA that we're sending
    //to remote processes. Have to send LCA labels because of local reductions.
    //std::cout<<"*******BEFORE COMM*********\n";
    communicate(g,verts_to_send, labels_to_send, procs_to_send, LCA_procs_to_send, LCA_labels, low_labels, remote_LCA_labels, remote_LCA_levels, levels,
		  curr_prop_queue, irreducible_prop_queue, irreducible_verts, potential_artpt_did_prop_lower,potential_artpts,ghost_offsets, ghost_adjs, did_recv_remote_LCA);

    //std::cout<<"*******AFTER COMM**********\n"; 
    //std::cout<<"Task "<<procid<<": done communicating... \n";
    //std::swap(curr_prop_queue, next_prop_queue);
    //std::cout<<"Task "<<procid<<": irreducible queue size = "<<irreducible_prop_queue->size()<<"\n";
    while(irreducible_prop_queue->size() > 0){
      curr_prop_queue->push(irreducible_prop_queue->front());
      irreducible_prop_queue->pop();
    }
    irreducible_verts.clear();
    //if all queues are empty, the loop can be broken
    int done = curr_prop_queue->size();
    global_done = 0;
    MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //while(true){}
  }
  
  /*std::ifstream known("arts");
  uint64_t* known_artpts = new uint64_t[g->n_total];
  for(int i = 0; i < g->n_total; i++) known_artpts[i] = 0;
  uint64_t art = 0;
  while(known>>art){
    known_artpts[art] = 1;
  } */

  int num_artpts = 0;
  //set articulation_point_flags for the caller.
  for(uint64_t i = 0; i < g->n_local; i++){
    if(g->local_unmap[i] == 140529) printf("Vertex %lu has LCA label %lu and low label %lu\n",g->local_unmap[i],*LCA_labels[i].begin(),low_labels[i]);

    /**
     * NOTE: I'm not entirely sure this translation of labels to articulation points holds in all cases.
     *       It seems like it works though, need to test more.
     *
     * TODO: A proof of correctness.
    */
    
    articulation_point_flags[i] = 0;
    if(potential_artpts[i] != 0 ){
      int out_degree = out_degree(g, i);
      uint64_t* nbors = out_vertices(g, i);
      for(int nbor = 0; nbor < out_degree; nbor++){
        if(levels[i] < levels[nbors[nbor]] && (LCA_labels[i] != LCA_labels[nbors[nbor]] || low_labels[i] != low_labels[nbors[nbor]])){
	  articulation_point_flags[i] = 1;
	  //if(procid == 2 && g->local_unmap[i] == 13) std::cout<<"13 is an artpt\n";
	  num_artpts++;
	  break;
	}
      }
    }
  }
  /*std::cout<<"num artpts: "<<num_artpts<<"\n";
  uint64_t false_negatives = 0;
  uint64_t false_positives = 0;
  uint64_t correct_answers = 0;
  for(int i = 0; i < g->n_total; i++){
    if(known_artpts[i] == 1 && articulation_point_flags[i] == 0) false_negatives++;
    if(known_artpts[i] == 0 && articulation_point_flags[i] == 1){
      false_positives++;
      print_labels(g,i,LCA_labels,low_labels,potential_artpts,levels);
      //printf("Vertex %lu has LCA label %lu and low label %lu\n",g->local_unmap[i],*LCA_labels[i].begin(),low_labels[i]);
    }
    if(known_artpts[i] == 1 && articulation_point_flags[i] == 1) correct_answers++;
  }
  std::cout<<"correct: "<<correct_answers<<" false_positives: "<<false_positives<<" false_negatives: "<<false_negatives<<"\n";*/
}


#endif

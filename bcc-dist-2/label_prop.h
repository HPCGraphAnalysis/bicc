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
   bool done = false;
   uint64_t curr_GID = curr_vtx;
   if(curr_vtx < g->n_local) curr_GID = g->local_unmap[curr_vtx];
   else curr_GID = g->ghost_unmap[curr_vtx-g->n_local];
  
   
    
   if(irreducible_verts.count(curr_GID) == 1){
     return false;
   }
   
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
       
       if(curr_level > highest_level){
         highest_level_gid = *it;
	 highest_level = curr_level;
       }
     }
     
     
     //we aren't done and we need to reduce the highest-level-valued label
     //remove the ID under consideration from the current label
     LCA_labels[curr_vtx].erase(highest_level_gid);
     std::set<uint64_t> labels_of_highest_label;
     if(get_value(g->map, highest_level_gid) == NULL_KEY){
       labels_of_highest_label = remote_LCA_labels[ highest_level_gid ];
     } else {
       labels_of_highest_label = LCA_labels[get_value(g->map, highest_level_gid)];
     }
     
     

     uint64_t level_of_labels_of_highest_label = 0;
     if(labels_of_highest_label.size() > 0 && get_value(g->map, *labels_of_highest_label.begin()) == NULL_KEY){
       level_of_labels_of_highest_label = remote_LCA_levels[*labels_of_highest_label.begin()];
     } else if( labels_of_highest_label.size() > 0){
       level_of_labels_of_highest_label = levels[get_value(g->map,*labels_of_highest_label.begin())];
     }
     
     if(labels_of_highest_label.count(curr_GID) == 1 && highest_level > levels[curr_vtx]){
       
       continue; 
     }
     
      
     
     
     if(labels_of_highest_label.size() == 1){
       //perform reduction successfully
       LCA_labels[curr_vtx].insert(*labels_of_highest_label.begin());    
       
     } else {
       LCA_labels[curr_vtx].insert(highest_level_gid);
       if(irreducible_verts.count(highest_level_gid) == 1){
         //this reduction is irreducible. Abort and add to irreducibility queue/set
         irreducible_verts.insert(curr_GID);
	 irreducible_prop_queue->push(curr_vtx);
	 return false;
       } else if(labels_of_highest_label.size() != 1 && get_value(g->map, highest_level_gid) >= g->n_local) {
         //the LCA which has no information is a ghost, so it is irreducible until communication. 
	 irreducible_verts.insert(curr_GID);
	 irreducible_prop_queue->push(curr_vtx);
	 return false;
       } else if(labels_of_highest_label.size() == 0){
	 //in distributed memory, impossible to tell wether this is actually going to be reducible without comm, so save for after comm.
         irreducible_verts.insert(curr_GID);
	 irreducible_prop_queue->push(curr_vtx);
	 return false;
       } else if(labels_of_highest_label.size() == 1 && highest_level < level_of_labels_of_highest_label){
         irreducible_verts.insert(curr_GID);
         irreducible_prop_queue->push(curr_vtx);
         return false;     
       } else { //irreducible_set doesn't contain the unreduced/empty LCA label, so it's local. just wait for it.
         //irreducible flag is not set, but there are multiple labels, cannot tell it's irreducible yet,
	 //put it back on the local prop queue and abort current reduction.
	 prop_queue->push(curr_vtx);
	 return false;
       }
     }
   }
   
   //if we get here, reduction was successful.
   return true;
}

//Just one at a time, please
void reduce_label(dist_graph_t* g, std::vector<uint64_t>& entry, std::queue<std::vector<uint64_t>>& redux_send_queue,
                  const std::vector<std::set<uint64_t>>& LCA_labels, const std::unordered_map<uint64_t,int>& LCA_owner, uint64_t* levels,
                  const std::unordered_map<uint64_t, int>& remote_levels){
  //Entry has the structure:
  //------------------------------------------------------------------------------------
  //|    |original | current|  num   | label_0|level of | owner of|                    |
  //|GID |owning   | owning | labels |        | label_0 | label_0 | (other labels here)|
  //|    |proc     | proc   |        |        |         |         |                    |
  //------------------------------------------------------------------------------------
  //0     1         2        3        4
  //find lowest LCA label in entry
  uint64_t lowest_level = entry[5];
  uint64_t lowest_LCA = entry[4];
  uint64_t lowest_owner = entry[6];
  uint64_t lowest_idx = 4;
  uint64_t num_labels = entry[3];
  for(int i = 4; i < 4+num_labels*3; i+= 3){
    if(entry[i+1] > lowest_level){
      lowest_idx = i;
      lowest_LCA = entry[i];
      lowest_level = entry[i+1];
      lowest_owner = entry[i+2];
    }
  }
  //if owned, replace that label with its label, if there is a single label.
  //          also ensure the replaced label does not already exist in the label. (no duplicates)
  if(lowest_owner == procid){
    uint64_t lowest_lid = get_value(g->map, lowest_LCA);
    if(LCA_labels.at(lowest_lid).size() == 1){
      entry.erase(entry.begin()+lowest_idx);
      entry.erase(entry.begin()+lowest_idx);
      entry.erase(entry.begin()+lowest_idx);
      uint64_t new_LCA = *LCA_labels.at(lowest_lid).begin();
      uint64_t new_level = 0;
      uint64_t new_owner = 0;
      if(get_value(g->map, new_LCA) != NULL_KEY){
        new_owner = procid;
        new_level = levels[get_value(g->map, new_LCA)];
      } else {
        new_owner = LCA_owner.at(new_LCA);
        new_level = remote_levels.at(new_LCA);
      }
      //only re-insert the new LCA if it doesn't already exist
      if(std::count(entry.begin()+4,entry.end(),new_LCA) == 0){
        entry.push_back(new_LCA);
        entry.push_back(new_level);
        entry.push_back(new_owner);
      } else{
        //need to reduce the number of labels present in the entry
        entry[3]--;
      }
    }
  } else {
  //if not owned, set the owner to whichever process owns the vertex in the label
    entry[2] = lowest_owner;
  }
  //put the entry on the send queue.
  redux_send_queue.push(entry);
}

void communicate_redux(dist_graph_t* g, std::queue<std::vector<uint64_t>>* redux_send_queue, std::queue<std::vector<uint64_t>>* next_redux_queue,
                       std::vector<std::set<uint64_t>>& LCA_labels, bool* potential_artpt_did_prop_lower, uint64_t* potential_artpts,
                       std::queue<uint64_t>* next_prop_queue, std::unordered_map<uint64_t,int>& LCA_owner, 
                       std::unordered_map<uint64_t, int>& remote_levels,
                       std::queue<std::vector<uint64_t>>* irredux_queue){
  //read each entry, send to the current owner.
  int* sendcnts = new int[nprocs];
  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendcnts[i] = 0;
    recvcnts[i] = 0;
  }
  int queue_size = redux_send_queue->size();
  for(int i = 0; i < queue_size; i++){
    std::vector<uint64_t> curr_entry = redux_send_queue->front();
    /*std::cout<<"sending entry with size "<<curr_entry.size()<<" to proc "<<curr_entry[2]<<"\n";
    std::cout<<"entry for vertex "<<curr_entry[0]<<" contains "<<curr_entry[3]<<" labels\n";*/
    if(curr_entry[0] == 191817){
      std::cout<<"sending entry for vertex "<<curr_entry[0]<<" with "<<curr_entry[3]<<" labels, with size "<<curr_entry.size()
               <<" starting at sendbuf idx "<<sendcnts[curr_entry[2]]<<"\n" ;
    }
    redux_send_queue->pop();
    sendcnts[curr_entry[2]] += curr_entry.size();
    redux_send_queue->push(curr_entry);
  } 

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
  int sendidx[nprocs];
  for(int i = 0; i < nprocs; i++) sendidx[i] = sdispls[i];
  
  while(redux_send_queue->size() > 0){
    std::vector<uint64_t> curr_entry = redux_send_queue->front();
    redux_send_queue->pop();
    //Entry has the structure:
    //------------------------------------------------------------------------------------
    //|    |original | current|  num   | label_0|level of | owner of|                    |
    //|GID |owning   | owning | labels |        | label_0 | label_0 | (other labels here)|
    //|    |proc     | proc   |        |        |         |         |                    |
    //------------------------------------------------------------------------------------
    //0     1         2        3        4
    int proc_to_send = curr_entry[2];
    int num_labels = curr_entry[3];
    int entry_size = num_labels*3+4;
    
    //std::cout<<"vertex "<<curr_entry[0]<<"\n";
    assert(entry_size == curr_entry.size());
    if(curr_entry[0] == 853019){
      //std::cout<<"inserting vertex "<<curr_entry[0]<<" at index "<<sendidx[proc_to_send]<<"\n";
    }
    for(int i = 0; i < entry_size; i++){
      sendbuf[sendidx[proc_to_send]++] = curr_entry[i];
    }
  }

  //MPI_Alltoallv
  MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);
  
  //on recv, update local labels for traversals that have finished (sent to this proc but multi-labeled and lowest owner is not procid)
  int ridx = 0;
  //std::cout<<"recvsize = "<<recvsize<<"\n";
  while(ridx < recvsize){
    //std::cout<<"ridx = "<<ridx<<"\n";
    //read in the current entry's info
    uint64_t gid = recvbuf[ridx];
    if(gid == 853019){
      //std::cout<<"comm_redux: processing recvbuf, vertex "<<gid<<"\n";
      //std::cout<<"\t has "<<recvbuf[ridx+3]<< " labels, at ridx "<<ridx<<" with recvsize "<<recvsize<<"\n";
    }
    uint64_t og_owner = recvbuf[ridx+1];
    uint64_t curr_owner = recvbuf[ridx+2];
    uint64_t num_labels = recvbuf[ridx+3];
    uint64_t entry_size = num_labels*3 +4;
    uint64_t lid = get_value(g->map, gid);

    if(num_labels == 1){
      //update label, and stop the traversal
      LCA_labels[lid].clear();
      uint64_t curr_label = recvbuf[ridx+4];
      uint64_t curr_label_level = recvbuf[ridx+4+1];
      uint64_t curr_label_owner = recvbuf[ridx+4+2];

      LCA_labels[lid].insert(curr_label);
      //if the LCA label is owned, we don't need to update any local info,
      //if it is NOT owned, mark the level and owner for future use.
      if(get_value(g->map, curr_label) == NULL_KEY){
        remote_levels[curr_label] = recvbuf[ridx+4+1];
        LCA_owner[curr_label] = recvbuf[ridx+4+2];
      }
      if(potential_artpts[lid]){
        potential_artpt_did_prop_lower[lid] = false;
      }
      if(lid == 816028) std::cout<<"Reduced vertex "<<lid<<"\n";
      next_prop_queue->push(lid); 
      //do nothing, this traversal is done.
    } else {
      //construct entry, and find owner of lowest LCA
      std::vector<uint64_t> entry;
      entry.push_back(gid);
      entry.push_back(og_owner);
      entry.push_back(curr_owner);
      entry.push_back(num_labels);
      uint64_t lowest_label = recvbuf[ridx+4];
      uint64_t lowest_level = recvbuf[ridx+4+1];
      uint64_t lowest_owner = recvbuf[ridx+4+2];
      //add the label info to the entry, we might put it back on the queue.
      //also, find the owner of the lowest label
      for(int i = 0; i < num_labels*3; i+=3){
        entry.push_back(recvbuf[ridx+4+i]);
        entry.push_back(recvbuf[ridx+4+i+1]);
        entry.push_back(recvbuf[ridx+4+i+2]);

        if(recvbuf[ridx+4+i+1] > lowest_level){
          lowest_label = recvbuf[ridx+4+i];
          lowest_level = recvbuf[ridx+4+i+1];
          lowest_owner = recvbuf[ridx+4+i+2];
        }
      }
      
      //On a traversal received, progress can only be made if (lowest_owner == procid && LCA_labels[lid].size() == 1)
      if(lowest_owner == procid && LCA_labels[get_value(g->map, lowest_label)].size() == 1){
        //put on the next_redux_queue
        next_redux_queue->push(entry);
      //if progress cannot be made, update the label and do not attempt to continue traversing
      } else{
        //update the label and put on irredux_queue
        LCA_labels[lid].clear();
        for(int i = 0; i < num_labels*3; i+=3){
          uint64_t lca_label = recvbuf[ridx+4+i];
          uint64_t lca_level = recvbuf[ridx+4+i+1];
          uint64_t lca_owner = recvbuf[ridx+4+i+2];
          LCA_labels[lid].insert(lca_label);
          if(get_value(g->map,lca_label) == NULL_KEY){
            LCA_owner[lca_label] = lca_owner;
            remote_levels[lca_label] = lca_level;
          }
        }
        irredux_queue->push(entry);
      }
    }
    /*if(procid == og_owner && og_owner == curr_owner){
      uint64_t lid = get_value(g->map, gid);
      std::vector<uint64_t> entry;
      entry.push_back(gid);
      entry.push_back(og_owner);
      entry.push_back(curr_owner);
      entry.push_back(num_labels);
      //std::cout<<"updating labels for vertex "<<lid<<"\n";
      LCA_labels[lid].clear();
      for(int i = 0; i < num_labels*3; i+=3){
        //std::cout<<"\tadding label "<<recvbuf[ridx+4+i]<<"\n";
        entry.push_back(recvbuf[ridx+4+i]);
        entry.push_back(recvbuf[ridx+4+i+1]);
        entry.push_back(recvbuf[ridx+4+i+2]);

        LCA_labels[lid].insert(recvbuf[ridx+4+i]);
      }
      if(num_labels == 1){
        //put vert on the next_prop_queue
        next_prop_queue->push(lid);
        if(potential_artpts[lid]){
          potential_artpt_did_prop_lower[lid] = false;
        }
      } else {
        //if the lowest vertex is not owned by this proc, do not put on next_redux_queue.
        //if 
        //next_redux_queue->push(entry);
      }
    } else {
      std::vector<uint64_t> entry;
      entry.push_back(gid);
      entry.push_back(og_owner);
      entry.push_back(curr_owner);
      entry.push_back(num_labels);
      for(int i = 0; i < num_labels*3; i+=1){
        entry.push_back(recvbuf[ridx+4+i]);
      }

      next_redux_queue->push(entry);
    }*/
    ridx += entry_size;
  }
}

//pass LCA and low labels between two neighboring vertices
void pass_labels(dist_graph_t* g,uint64_t curr_vtx, uint64_t nbor, std::vector<std::set<uint64_t>>& LCA_labels,
		 std::unordered_map<uint64_t, int>& remote_levels,
		 uint64_t* low_labels, uint64_t* levels, uint64_t* potential_artpts, bool* potential_artpt_did_prop_lower,
		 std::queue<uint64_t>* prop_queue, std::queue<uint64_t>* next_prop_queue, std::set<uint64_t>& verts_to_send,
		 std::unordered_map<uint64_t, std::set<int>>& procs_to_send,
		 bool full_reduce, bool reduction_needed){
  if(curr_vtx == 816028 && nbor == 815634){
    std::cout<<curr_vtx<<" passing labels to "<<nbor<<"\n";
    std::cout<<"full_reduce = "<<full_reduce<<" reduction_needed = "<<reduction_needed<<"\n";
    std::cout<<"LCA_labels[curr_vtx].size() = "<<LCA_labels[curr_vtx].size()<<"\n";
    std::cout<<"LCA_labels[curr_vtx]: ";
    for(auto it = LCA_labels[curr_vtx].begin(); it != LCA_labels[curr_vtx].end(); it++){
      std::cout<<*it<<" ";
    }
    std::cout<<"\n";
  }
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
      }
    }
    //pass low_label to neighbor if LCA_labels are the same, and if it is lower.
    uint64_t curr_gid = curr_vtx;
    if(curr_vtx < g->n_local) curr_gid = g->local_unmap[curr_vtx];
    else curr_gid = g->ghost_unmap[curr_vtx-g->n_local];

    if(LCA_labels[curr_vtx] == LCA_labels[nbor] &&
		    (levels[nbor] <= levels[curr_vtx] || *LCA_labels[nbor].begin() != curr_gid)){
      uint64_t curr_low_label = low_labels[curr_vtx];
      uint64_t nbor_low_label = low_labels[nbor];
      uint64_t curr_low_label_level = 0;
      if(get_value(g->map, curr_low_label) == NULL_KEY){
        curr_low_label_level = remote_levels[curr_low_label];
      } else {
        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
      }
      uint64_t nbor_low_label_level = 0;
      if(get_value(g->map, nbor_low_label) == NULL_KEY){
        nbor_low_label_level = remote_levels[nbor_low_label];
      } else {
        nbor_low_label_level = levels[get_value(g->map,nbor_low_label)];
      }
      
      if(curr_low_label_level > nbor_low_label_level ||
		      (curr_low_label_level == nbor_low_label_level && curr_low_label > nbor_low_label)){
        
        low_labels[nbor] = low_labels[curr_vtx];
        nbor_changed = true;
      }
    }
  } else {
    //for non-LCA verts, only pass labels if
    if((levels[nbor] > levels[curr_vtx] ) || // the neighbor is lower and has no labels, or
       (levels[nbor] <= levels[curr_vtx] && full_reduce )){ //the neighbor is higher and the current label was recently reduced.
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
	          level_of_diff = remote_levels[diff[i]];
	        } else {
	          level_of_diff = levels[get_value(g->map, diff[i])];
	        }
          if(diff[i] != nbor_gid && level_of_diff < levels[nbor]) {
            LCA_labels[nbor].insert(diff[i]);
            nbor_changed = true;
          }
        }
      }
    }
    
    if(LCA_labels[curr_vtx] == LCA_labels[nbor]){
      uint64_t curr_low_label = low_labels[curr_vtx];
      uint64_t nbor_low_label = low_labels[nbor];
      uint64_t curr_low_label_level = 0;
      if(get_value(g->map, curr_low_label) == NULL_KEY){
        curr_low_label_level = remote_levels[curr_low_label];
      } else {
        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
      }
      uint64_t nbor_low_label_level = 0;
      if(get_value(g->map, nbor_low_label) == NULL_KEY){
        nbor_low_label_level = remote_levels[nbor_low_label];
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
    //add it to verts_to_send.

    if(procs_to_send[nbor].size() > 0){
      verts_to_send.insert(nbor);
    }
    next_prop_queue->push(nbor);
  }
}

void communicate_prop(dist_graph_t* g, std::set<uint64_t> verts_to_send, std::unordered_map<uint64_t, std::set<int>>& procs_to_send,
                      std::vector<std::set<uint64_t>>& LCA_labels, uint64_t* low_labels, std::unordered_map<uint64_t, int>& LCA_owner,
                      uint64_t* levels, std::unordered_map<uint64_t, int>& remote_levels, std::queue<uint64_t>* next_prop_queue){
  bool already_sent[g->n_local];
  int* sendcnts = new int[nprocs];
  int* recvcnts = new int[nprocs];
  for(int i = 0; i < nprocs; i++){
    sendcnts[i] = 0;
    recvcnts[i] = 0;
  }
  for(int i = 0; i < g->n_local; i++) already_sent[i] = false;
  for(auto vtxit = verts_to_send.begin(); vtxit != verts_to_send.end(); vtxit++){
    uint64_t curr_vtx = *vtxit;
    if(LCA_labels[curr_vtx].size() == 1 && !already_sent[curr_vtx]){
      for(auto it = procs_to_send[curr_vtx].begin(); it != procs_to_send[curr_vtx].end(); ++it){
        sendcnts[*it] += 6;
      }
      already_sent[curr_vtx] = true;
    }
  }
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
  int sendidx[nprocs];
  for(int i = 0; i < nprocs; i++) sendidx[i] = sdispls[i];
  for(int i = 0; i < g->n_local; i++) already_sent[i] = false;
 
  for(auto it = verts_to_send.begin(); it != verts_to_send.end(); it++){
    if(LCA_labels[*it].size() == 1 && !already_sent[*it]){
      uint64_t LCA_gid = *LCA_labels[*it].begin();
      int lca_level = 0;
      int lca_owner = procid;
      int low_level = 0;
      if(get_value(g->map, LCA_gid) == NULL_KEY){
        lca_level = remote_levels[LCA_gid];
        lca_owner = LCA_owner[LCA_gid];
      } else {
        lca_level = levels[get_value(g->map, LCA_gid)];
      }
      if(get_value(g->map, low_labels[*it]) == NULL_KEY){
        low_level = remote_levels[low_labels[*it]];
      } else {
        low_level = levels[low_labels[*it]];
      }
      for(auto p_it = procs_to_send[*it].begin(); p_it != procs_to_send[*it].end(); ++p_it){
        sendbuf[sendidx[*p_it]++] = g->local_unmap[*it];
        sendbuf[sendidx[*p_it]++] = LCA_gid;
        sendbuf[sendidx[*p_it]++] = lca_level;
        sendbuf[sendidx[*p_it]++] = lca_owner;
        sendbuf[sendidx[*p_it]++] = low_labels[*it];
        sendbuf[sendidx[*p_it]++] = low_level;
      }
      already_sent[*it] = true;
    }
  }

  MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);

  for(int i = 0; i < recvsize; i+=6){
    uint64_t lid = get_value(g->map,recvbuf[i]);//should always be a ghost. No completely remote vertices here.
    uint64_t lca_gid = recvbuf[i+1]; //might be a remote LCA, need to set the owner
    uint64_t lca_level = recvbuf[i+2];
    uint64_t lca_owner = recvbuf[i+3];
    uint64_t low_label = recvbuf[i+4];
    uint64_t low_level = recvbuf[i+5];
    LCA_labels[lid].clear();
    LCA_labels[lid].insert(lca_gid);
    low_labels[lid] = low_label;
    //if LCA is not owned, add to LCA_owner
    //same with remote_levels
    if(get_value(g->map,lca_gid) == NULL_KEY){
      LCA_owner[lca_gid] = lca_owner;
      remote_levels[lca_gid] = lca_level;
    }
    //if low label is not owned, add level to remote_levels
    if(get_value(g->map,low_label) == NULL_KEY){
      remote_levels[low_label] = low_level;
    }
  }
  delete [] sendcnts;
  delete [] recvcnts;
}
void pull_low_labels(uint64_t curr_vtx, uint64_t nbor, dist_graph_t* g, std::vector<uint64_t>& ghost_offsets, std::vector<uint64_t>& ghost_adjs,
                     std::vector<std::set<uint64_t>>& LCA_labels, uint64_t* potential_artpts, uint64_t* low_labels, std::set<uint64_t>& verts_to_send, 
                     std::unordered_map<uint64_t, std::set<int>>& procs_to_send, uint64_t* levels, std::unordered_map<uint64_t,int>& remote_levels){
  int out_degree = 0;
  uint64_t* nbors = nullptr;
  if(curr_vtx < g->n_local){
    out_degree = out_degree(g, curr_vtx);
    nbors = out_vertices(g,curr_vtx);
  } else {
    out_degree = ghost_offsets[curr_vtx+1 - g->n_local] - ghost_offsets[curr_vtx - g->n_local];
    nbors = &ghost_adjs[ghost_offsets[curr_vtx-g->n_local]];
  }

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
        curr_low_label_level = remote_levels.at(curr_low_label);
      } else {
        curr_low_label_level = levels[get_value(g->map, curr_low_label)];
      }
      uint64_t nbor_low_label_level = 0;
      if(get_value(g->map, nbor_low_label) == NULL_KEY){
        nbor_low_label_level = remote_levels.at(nbor_low_label);
      } else {
        nbor_low_label_level = levels[get_value(g->map, nbor_low_label)];
      }

      if(curr_low_label_level < nbor_low_label_level ||
          (curr_low_label_level == nbor_low_label_level && curr_low_label < nbor_low_label)){
        low_labels[curr_vtx] = low_labels[nbor];
        if(procs_to_send[curr_vtx].size() > 0) verts_to_send.insert(curr_vtx);
      }
    }
  }
}

void print_labels(dist_graph_t *g, uint64_t vertex, std::vector<std::set<uint64_t>> LCA_labels, uint64_t* low_labels, uint64_t* potential_artpts, uint64_t* levels){

  if(vertex < g->n_local) std::cout<<"Task "<<procid<<": vertex "<<g->local_unmap[vertex]<<" has LCA label "<<*LCA_labels[vertex].begin()<<", low label "<<low_labels[vertex]<<" and level "<<levels[vertex];
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
  //keep track of ghosts we need to send
  std::set<uint64_t> verts_to_send;
  
  //keep track of propagation
  std::queue<uint64_t> prop_queue;
  std::queue<uint64_t> n_prop_queue;

  //keep track of reduction
  std::queue<std::vector<uint64_t>> reduction_queue;
  std::queue<std::vector<uint64_t>> n_reduction_queue;
  std::queue<std::vector<uint64_t>> irreducible_queue;
  std::queue<std::vector<uint64_t>> reduction_to_send_queue;
  //need to know LCA owners, levels
  std::unordered_map<uint64_t, int> LCA_owner;
  std::unordered_map<uint64_t, int> remote_levels;

  //aliases for easy queue switching
  std::queue<uint64_t> * curr_prop_queue = &prop_queue;
  std::queue<uint64_t> * next_prop_queue = &n_prop_queue;
  std::queue<std::vector<uint64_t>>* curr_redux_queue = &reduction_queue;
  std::queue<std::vector<uint64_t>>* next_redux_queue = &n_reduction_queue;
  std::queue<std::vector<uint64_t>>* irredux_queue = &irreducible_queue;
  std::queue<std::vector<uint64_t>>* redux_to_send_queue = &reduction_to_send_queue;
  
  bool* potential_artpt_did_prop_lower = new bool[g->n_total];
  bool* did_recv_remote_LCA = new bool[g->n];
  for(uint64_t i = 0; i < g->n; i++) did_recv_remote_LCA[i] = false;
  std::set<uint64_t> irreducible_verts;
  double total_comm_time = 0.0;
  int total_send_size = 0;

  for(uint64_t i = 0; i < g->n_total; i++){
    if(potential_artpts[i] != 0) {
      curr_prop_queue->push(i);
      if(procs_to_send[i].size() > 0){
        verts_to_send.insert(i);
      }
    }
    potential_artpt_did_prop_lower[i] = false;
    if(levels[i] == 0) {
      if(i < g->n_local) LCA_labels[i].insert(g->local_unmap[i]);
      else LCA_labels[i].insert(g->ghost_unmap[i-g->n_local]);
      if(procs_to_send[i].size() > 0){
        verts_to_send.insert(i);
      }
    }
  }
  
  uint64_t done = curr_prop_queue->size();
  uint64_t global_done = 0;
  MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  bool prop_till_done = true;
  bool redux_till_done = true;
  bool reduction_needed[g->n_total];
  for(int i = 0; i < g->n_total; i++) reduction_needed[i] = false; 
  //continue propagating and reducing until the propagations and reductions are all completed.
  while( global_done > 0){
    
    //check the size of the global queue, propagate until the current queue is empty.
    uint64_t global_prop_size = 0;
    uint64_t local_prop_size = curr_prop_queue->size();
    MPI_Allreduce(&local_prop_size, &global_prop_size,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    std::cout<<"Passing labels\n";
    //prop_till_done lets us propagate until no further propagation is possible,
    //otherwise it just propagates until the current queue is empty.
    while((prop_till_done && global_prop_size > 0) || (!prop_till_done && curr_prop_queue->size() > 0)){
      //go through curr_prop_queue and propagate, filling next_prop_queue with everything that propagates next.
      uint64_t curr_vtx = curr_prop_queue->front();
      curr_prop_queue->pop();
      uint64_t curr_gid = curr_vtx;
      if(curr_vtx < g->n_local) curr_gid = g->local_unmap[curr_vtx];
      else curr_gid = g->ghost_unmap[curr_vtx - g->n_local];

      int out_degree = 0;
      uint64_t* nbors = nullptr;
      if(curr_vtx < g->n_local){
        out_degree = out_degree(g, curr_vtx);
        nbors = out_vertices(g, curr_vtx);
      } else {
        out_degree = ghost_offsets[curr_vtx+1 - g->n_local] - ghost_offsets[curr_vtx - g->n_local];
        nbors = &ghost_adjs[ghost_offsets[curr_vtx-g->n_local]];
      }
      bool full_reduce = LCA_labels[curr_vtx].size() == 1;
      //bool reduction_needed = false;
      for(int nbor_idx = 0; nbor_idx < out_degree; nbor_idx++){
        if(full_reduce && reduction_needed[curr_vtx]){
          pull_low_labels(curr_vtx, nbors[nbor_idx],g,ghost_offsets,ghost_adjs, LCA_labels,potential_artpts,low_labels,
                          verts_to_send,procs_to_send,levels,remote_levels);
        }
        if(nbors[nbor_idx] == 1){
          /*std::cout<<"before vertex "<<curr_vtx<<"passes, vertex 1 has labels:\n";
          for(auto it = LCA_labels[1].begin(); it != LCA_labels[1].end(); it++){
            std::cout<<*it<<" ";
          }
          std::cout<<"\n";*/
        }
        pass_labels(g,curr_vtx,nbors[nbor_idx],LCA_labels, remote_levels, low_labels, levels, potential_artpts,
                    potential_artpt_did_prop_lower, curr_prop_queue, next_prop_queue, verts_to_send, procs_to_send,
                    full_reduce, reduction_needed[curr_vtx]);
        if(nbors[nbor_idx] == 1){
          /*std::cout<<"after vertex "<<curr_vtx<<"passes, vertex 1 has labels:\n";
          for(auto it = LCA_labels[1].begin(); it != LCA_labels[1].end(); it++){
            std::cout<<*it<<" ";
          }
          std::cout<<"\n";*/
        }
      }
      if(potential_artpts[curr_vtx] && !potential_artpt_did_prop_lower[curr_vtx]){
        potential_artpt_did_prop_lower[curr_vtx] = true;
      } 
      //if prop_till_done is set, we need to update the global_prop_size with next_prop_queue->size()
      //and swap next_prop_queue with curr_prop_queue.
      if(prop_till_done && curr_prop_queue->size() == 0){
        communicate_prop(g,verts_to_send,procs_to_send,LCA_labels,low_labels,LCA_owner,levels,remote_levels,next_prop_queue);
        std::swap(curr_prop_queue,next_prop_queue);
        local_prop_size = next_prop_queue->size();
        global_prop_size = 0;
        MPI_Allreduce(&local_prop_size,&global_prop_size,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      }
    }
    std::cout<<"Done propagating\n";
    if(!prop_till_done){
      communicate_prop(g,verts_to_send,procs_to_send,LCA_labels,low_labels,LCA_owner,levels,remote_levels,next_prop_queue);
      std::swap(curr_prop_queue, next_prop_queue);
    }
    while(irredux_queue->size()) irredux_queue->pop();
    std::cout<<"Finding vertices in need of reduction\n";
    for(int i = 0; i < g->n_local; i++){
      if(LCA_labels[i].size() > 1){
        reduction_needed[i] = true;
        //entries in the reduction queue are of the form
        //-----------------------------------------------------------------------------------------------------
        //| GID | owner |current_owner| #labels | label_0, level_0, owner_0 | ... | label_n, level_n, owner_n |
        //-----------------------------------------------------------------------------------------------------
        std::vector<uint64_t> temp_entry;
        temp_entry.push_back(g->local_unmap[i]);//GID
        temp_entry.push_back(procid);//original owner of vertex (generated this entry)
        temp_entry.push_back(procid);//current owner (currently has the entry)
        temp_entry.push_back(LCA_labels[i].size());//number of labels (3*this number is the size of this entry)
        for(auto it = LCA_labels[i].begin(); it != LCA_labels[i].end(); it++){
          temp_entry.push_back(*it);
          //std::cout<<"vertex "<<i<<" pushing back label "<<*it<<"\n";
          if(get_value(g->map,*it) == NULL_KEY){
            temp_entry.push_back(remote_levels[*it]);
          } else {
            temp_entry.push_back(levels[get_value(g->map,*it)]);
          }
          temp_entry.push_back(LCA_owner[*it]);
        }
        /*if(i == 1){
          std::cout<<"entry for vertex 1 is size "<<temp_entry.size()<<" with "<<LCA_labels[i].size()<<" labels, should be "<<4+LCA_labels[i].size()*3<<"\n";
        }*/
        curr_redux_queue->push(temp_entry);
      } else {
        reduction_needed[i] = false;
      }
    }
    std::cout<<"Done finding vertices in need of reduction\n";
    
    uint64_t local_redux_size = curr_redux_queue->size();
    uint64_t global_redux_size = 0;
    MPI_Allreduce(&local_redux_size,&global_redux_size,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    std::queue<std::vector<uint64_t>> redux_send_queue;
    std::cout<<"Reducing labels\n";
    while((redux_till_done && global_redux_size > 0) || (!redux_till_done && curr_redux_queue->size() > 0)){
      std::vector<uint64_t> curr_reduction = curr_redux_queue->front();
      curr_redux_queue->pop();
      /*if(curr_redux_queue->size() == 42019){
        std::cout<<" first reduction has gid "<<curr_reduction[0]<<"\n";
      }*/
      /*if(curr_reduction[0] == 243){
        std::cout<<"entry size = "<<curr_reduction.size()<<"\n";
        std::cout<<"before reduction, vertex "<<curr_reduction[0]<<" has labels:\n";
        for(auto it = LCA_labels[curr_reduction[0]].begin(); it != LCA_labels[curr_reduction[0]].end(); it++){
          std::cout<<*it<<" ";
        }
        std::cout<<"\n";
      }*/
      reduce_label(g, curr_reduction,redux_send_queue, LCA_labels, LCA_owner, levels, remote_levels); 
      /*if(curr_reduction[0] == 243){
        std::cout<<"after reduction, vertex "<<curr_reduction[0]<<" has labels:\n";
        for(auto it = LCA_labels[curr_reduction[0]].begin(); it != LCA_labels[curr_reduction[0]].end(); it++){
          std::cout<<*it<<" (level "<<levels[*it]<<", has "<<LCA_labels[*it].size()<<" labels) ";
        }
        std::cout<<"\n";
      }*/


      //TODO: Make sure that fully reduced LCA vertices (LCAs that had multiple labels but now only have 1)
      //      get potential_artpt_did_prop_lower set to true on their owning process.
      //reduce labels:
      //  - do local reductions
      //  - send reductions whose lowest level LCA is owned by another process to that process
      //  - send reductions for a vertex owned by another process back to the owner if progress cannot be made
      //  - update local labels
      //  - repeat until no progress can be made.
      if(redux_till_done && curr_redux_queue->size() == 0){
        //std::cout<<"communicating reductions\n";
        communicate_redux(g,&redux_send_queue,next_redux_queue,LCA_labels, potential_artpt_did_prop_lower,
                               potential_artpts, curr_prop_queue,LCA_owner, remote_levels,irredux_queue);
        //std::cout<<"done communicating reductions, next_redux_queue->size() = "<<next_redux_queue->size()<<"\n";
        /*std::cout<<"after communicating, vertex 1 has labels:\n";
        for(auto it = LCA_labels[1].begin(); it != LCA_labels[1].end(); it++){
          std::cout<<*it<<" ";
        }
        std::cout<<"\n";*/
        std::swap(curr_redux_queue,next_redux_queue);
        local_redux_size = curr_redux_queue->size();
        global_redux_size = 0;
        MPI_Allreduce(&local_redux_size, &global_redux_size,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      }
    }
    if(!redux_till_done){
      //communicate reductions
      communicate_redux(g,&redux_send_queue, next_redux_queue, LCA_labels, potential_artpt_did_prop_lower,
                             potential_artpts, next_prop_queue,LCA_owner, remote_levels,irredux_queue);
      std::swap(curr_redux_queue, next_redux_queue);
    }

    done = curr_prop_queue->size() + next_prop_queue->size() + curr_redux_queue->size() + next_redux_queue->size() + irredux_queue->size();
    MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //std::cout<<"global done: "<<global_done<<"\n";
  }
  std::cout<<"Done reducing labels\n";
  //set the return values
  int num_artpts = 0;
  int num_unreduced = 0;
  for(uint64_t i = 0; i < g->n_local; i++){
    if(LCA_labels[i].size() > 1) num_unreduced ++;
    articulation_point_flags[i] = 0;
    if(potential_artpts[i] != 0){
      int out_degree = out_degree(g,i);
      uint64_t* nbors = out_vertices(g, i);
      for(int nbor = 0; nbor < out_degree; nbor++){
        if(levels[i] < levels[nbors[nbor]] && (LCA_labels[i] != LCA_labels[nbors[nbor]] || low_labels[i] != low_labels[nbors[nbor]])){
          articulation_point_flags[i] = 1;
          num_artpts++;
          break;
        }
      }
    }
  }
  //std::cout<<"there are "<<num_unreduced<<" unreduced labels\n";
  print_labels(g, 816028, LCA_labels, low_labels, potential_artpts, levels);
  /*int global_send_total = 0;
  MPI_Allreduce(&total_send_size, &global_send_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(procid == 0){
    std::cout<<"Comm Time: "<<total_comm_time<<"\n";
    std::cout<<"Total send size: "<<global_send_total<<"\n";
    std::cout<<"Num comms: "<<num_comms<<"\n";
  }*/
}


#endif

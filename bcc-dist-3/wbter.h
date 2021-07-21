#ifndef _WBTER_H_
#define _WBTER_H_

#include <cstdint>

struct xs1024star_t {
  uint64_t s[16];
  int64_t p;
} ;

uint64_t xs1024star_next(xs1024star_t* xs);

double xs1024star_next_real(xs1024star_t* xs);

void xs1024star_seed(uint64_t seed, xs1024star_t* xs);

void xs1024star_seed(xs1024star_t* xs);

int read_nd_cd(char* nd_filename, char* cd_filename, 
  uint64_t*& nd, double*& cd, 
  uint64_t& num_verts, uint64_t& num_edges, uint64_t& dmax);

int initialize_bter_data(
  uint64_t* nd, double* cd, 
  uint64_t dmax, uint64_t& num_verts,
  uint64_t* id, double*& wd, double* rdfill, uint64_t* ndfill, 
  double*& wg, uint64_t* ig, uint64_t* bg, uint64_t* ng, double* pg, 
  uint64_t* ndprime, uint64_t& num_groups, uint64_t& num_blocks,
  double& w1, double& w2);

int calculate_parallel_distribution(
  uint64_t* nd, double* wd,
  uint64_t* ig, uint64_t* bg, uint64_t*& bg_ps, uint64_t* ng, double* pg, 
  uint64_t& start_p1, uint64_t& end_p1, uint64_t& start_p2, uint64_t& end_p2,
  uint64_t& my_gen_edges, uint64_t num_groups, uint64_t num_blocks, double w2);

int generate_bter_edges(uint64_t dmax, uint64_t* gen_edges, 
  uint64_t* id, uint64_t* nd, double* wd, double* rdfill, uint64_t* ndfill,
  uint64_t* ig, uint64_t* bg, uint64_t* bg_ps, uint64_t* ng, double* pg, 
  uint64_t start_p1, uint64_t end_p1, uint64_t start_p2, uint64_t end_p2,
  uint64_t& phase1_edges, uint64_t& phase2_edges, uint64_t& phase3_edges,
  uint64_t& num_gen_edges,  uint64_t num_groups, double w2);

int generate_bter_MPI(char* nd_filename, char* cd_filename, 
  uint64_t*& gen_edges, int64_t*& ground_truth,
  uint64_t& num_local_edges, uint64_t& num_local_verts, uint64_t& local_offset,
  bool remove_zeros);

int exchange_edges(uint64_t* nd, uint64_t dmax, uint64_t* id,
  uint64_t*& gen_edges, uint64_t& num_edges, uint64_t& num_gen_edges, 
  uint64_t num_verts, uint64_t& local_offset);

int assign_ground_truth(uint64_t* bg, uint64_t* ng, uint64_t& num_groups,
   uint64_t num_verts, int64_t*& ground_truth);

int assign_ground_truth_no_zeros(uint64_t* nd, uint64_t dmax, 
  uint64_t* id, uint64_t* bg, uint64_t* ng, uint64_t& num_groups,
  uint64_t& num_gen_edges, uint64_t num_verts, int64_t*& ground_truth);

int remove_zero_degrees(uint64_t* nd, uint64_t dmax, uint64_t* id,
  uint64_t*& gen_edges, uint64_t& num_edges, uint64_t& num_gen_edges, 
  uint64_t& num_verts, int64_t*& ground_truth,
  uint64_t& num_local_verts, uint64_t& num_local_edges);

void parallel_prefixsums(
  uint64_t* in_array, uint64_t* out_array, uint64_t size);

void parallel_prefixsums(double* in_array, double* out_array, uint64_t size);

int32_t binary_search(uint64_t* array, uint64_t value, int32_t max_index);

uint64_t binary_search(double* array, double value, uint64_t max_index);

uint64_t binary_search(uint64_t* array, uint64_t value, uint64_t max_index);

#endif
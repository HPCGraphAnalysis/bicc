
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <vector>
#include <fstream>

#include "wbter.h"
#include "util.h"

#define THREAD_QUEUE_SIZE 512
#define MAX_SEND_SIZE 2147483648
#define min(a, b) a < b ? a : b
//#define MANYCORE 1

extern int procid, nprocs;
extern bool verbose, debug;


uint64_t xs1024star_next(xs1024star_t* xs) 
{
   const uint64_t s0 = xs->s[xs->p];
   uint64_t s1 = xs->s[xs->p = (xs->p + 1) & 15];
   s1 ^= s1 << 31;
   xs->s[xs->p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30);
   return xs->s[xs->p] * uint64_t(1181783497276652981U);
}

double xs1024star_next_real(xs1024star_t* xs) 
{
   const uint64_t s0 = xs->s[xs->p];
   uint64_t s1 = xs->s[xs->p = (xs->p + 1) & 15];
   s1 ^= s1 << 31;
   xs->s[xs->p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30);
   double ret = (double)(xs->s[xs->p] * uint64_t(1181783497276652981U));   
   return ret /= (double)uint64_t(18446744073709551615U);
}

void xs1024star_seed(uint64_t seed, xs1024star_t* xs) 
{
  for (uint64_t i = 0; i < 16; ++i)
  {
    uint64_t z = (seed += uint64_t(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * uint64_t(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * uint64_t(0x94D049BB133111EB);
    xs->s[i] = z ^ (z >> 31);
  }
  xs->p = 0;
}

void xs1024star_seed(xs1024star_t* xs) 
{
  uint64_t seed = (uint64_t)rand();
  
  for (uint64_t i = 0; i < 16; ++i)
  {
    uint64_t z = (seed += uint64_t(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * uint64_t(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * uint64_t(0x94D049BB133111EB);
    xs->s[i] = z ^ (z >> 31);
  }
  xs->p = 0;
}

// void parallel_prefixsums(
//   uint64_t* in_array, uint64_t* out_array, uint64_t size)
// {
//   uint64_t* thread_sums;

// #pragma omp parallel
// {
//   uint64_t nthreads = (uint64_t)omp_get_num_threads();
//   uint64_t tid = (uint64_t)omp_get_thread_num();
// #pragma omp single
// {
//   thread_sums = new uint64_t[nthreads+1];
//   thread_sums[0] = 0;
// }

//   uint64_t my_sum = 0;
// #pragma omp for schedule(static)
//   for(uint64_t i = 0; i < size; ++i) {
//     my_sum += in_array[i];
//     out_array[i] = my_sum;
//   }

//   thread_sums[tid+1] = my_sum;
// #pragma omp barrier

//   uint64_t my_offset = 0;
//   for(uint64_t i = 0; i < (tid+1); ++i)
//     my_offset += thread_sums[i];

// #pragma omp for schedule(static)
//   for(uint64_t i = 0; i < size; ++i)
//     out_array[i] += my_offset;
// }

//   delete [] thread_sums;
// }

void parallel_prefixsums(double* in_array, double* out_array, uint64_t size)
{
  double* thread_sums;

#pragma omp parallel
{
  uint64_t nthreads = (uint64_t)omp_get_num_threads();
  uint64_t tid = (uint64_t)omp_get_thread_num();
#pragma omp single
{
  thread_sums = new double[nthreads+1];
  thread_sums[0] = 0;
}

  double my_sum = 0;
#pragma omp for schedule(static)
  for(uint64_t i = 0; i < size; ++i) {
    my_sum += in_array[i];
    out_array[i] = my_sum;
  }

  thread_sums[tid+1] = my_sum;
#pragma omp barrier

  double my_offset = 0;
  for(uint64_t i = 0; i < (tid+1); ++i)
    my_offset += thread_sums[i];

#pragma omp for schedule(static)
  for(uint64_t i = 0; i < size; ++i)
    out_array[i] += my_offset;
}

  delete [] thread_sums;
}


int32_t binary_search(uint64_t* array, uint64_t value, int32_t max_index)
{
  bool found = false;
  int32_t index = 0;
  int32_t bound_low = 0;
  int32_t bound_high = max_index;
  while (!found)
  {
    index = (bound_high + bound_low) / 2;
    if (array[index] <= value && array[index+1] > value)
    {
      return index;
    }
    else if (array[index] <= value)
      bound_low = index;
    else if (array[index] > value)
      bound_high = index;
  }

  return index;
}

uint64_t binary_search(double* array, double value, uint64_t max_index)
{
  bool found = false;
  uint64_t index = 0;
  uint64_t bound_low = 0;
  uint64_t bound_high = max_index;
  while (!found)
  {
    index = (bound_high + bound_low) / 2;
    if (array[index] <= value && array[index+1] > value)
    {
      return index;
    }
    else if (array[index] <= value)
      bound_low = index;
    else if (array[index] > value)
      bound_high = index;
  }

  return index;
}

uint64_t binary_search(uint64_t* array, uint64_t value, uint64_t max_index)
{
  bool found = false;
  uint64_t index = 0;
  uint64_t bound_low = 0;
  uint64_t bound_high = max_index;
  while (!found)
  {
    index = (bound_high + bound_low) / 2;
    if (array[index] <= value && array[index+1] > value)
    {
      return index;
    }
    else if (array[index] <= value)
      bound_low = index;
    else if (array[index] > value)
      bound_high = index;
  }

  return index;
}

int read_nd_cd(char* nd_filename, char* cd_filename, 
  uint64_t*& nd, double*& cd, 
  uint64_t& num_verts, uint64_t& num_edges, uint64_t& dmax)
{
  double elt = omp_get_wtime();
  if (debug) {
    printf("%d -- read_nd_cd() start ... \n", procid);
  }

  if (procid == 0) {
    std::ifstream infile;
    std::string line;

    dmax = 0;
    uint64_t alloc_size = 1024;
    nd = (uint64_t*)malloc(alloc_size*sizeof(uint64_t));
    nd[0] = 0;
    num_verts = 0;
    num_edges = 0;

    infile.open(nd_filename);
    while(getline(infile, line)) {
      ++dmax;
      if (dmax >= alloc_size) {
        nd = (uint64_t*)realloc(nd, alloc_size*2*sizeof(uint64_t));
        alloc_size *= 2;
      }

      uint64_t num = strtoul(line.c_str(), NULL, 0);
      nd[dmax] = num;
      num_verts += num;
      num_edges += (long)num*(long)dmax;
    }
    infile.close();

    cd = new double[dmax+1];
    cd[0] = 0.0;
    uint64_t cur_degree = 0;

    infile.open(cd_filename);
    while(getline(infile, line)) {   
      ++cur_degree;

      double cc = atof(line.c_str());
      cd[cur_degree] = cc;
    }
    infile.close();

    assert(cur_degree == dmax);
    num_edges /= 2;

    MPI_Bcast(&num_verts, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dmax, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    MPI_Bcast(nd, (dmax+1), MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(cd, (dmax+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(&num_verts, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dmax, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    nd = (uint64_t*)malloc((dmax+1)*sizeof(uint64_t));
    cd = (double*)malloc((dmax+1)*sizeof(double));

    MPI_Bcast(nd, (dmax+1), MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(cd, (dmax+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if (verbose && procid == 0) {
    printf("Desired num verts: %lu\n", num_verts);
    printf("Desired num edges: %lu\n", num_edges);
    printf("Desired max degree: %lu\n", dmax);
  }

  if (debug) {
    printf("%d -- read_nd_cd() done: %lf (s)\n", procid, omp_get_wtime() - elt);
  }

  return 0;
}

int initialize_bter_data(
  uint64_t* nd, double* cd, 
  uint64_t dmax, uint64_t& num_verts,
  uint64_t* id, double*& wd, double* rdfill, uint64_t* ndfill, 
  double*& wg, uint64_t* ig, uint64_t* bg, uint64_t* ng, double* pg, 
  uint64_t* ndprime, uint64_t& num_groups, uint64_t& num_blocks,
  double& w1, double& w2)
{
  double elt = omp_get_wtime();
  if (debug) {
    printf("%d -- initialize_bter_data() start ... \n", procid);
  }

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) id[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) wd[i] = 0.0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) rdfill[i] = 0.0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) ndfill[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) wg[i] = 0.0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) ig[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) bg[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) ng[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) pg[i] = 0.0;
#pragma omp for nowait
  for (uint64_t i = 0; i <= dmax; ++i) ndprime[i] = 0;
}
  if (verbose && procid == 0) printf(".");

  // Index of first node for each degree. 
  // Degree 1 vertices are numbered last.
  id[2] = 0;
  for (uint64_t d = 3; d <= dmax; ++d)
    id[d] = id[d-1] + nd[d-1];
  id[1] = id[dmax] + nd[dmax];
  if (verbose && procid == 0) printf(".");

  // Compute number of nodes with degree greater than d
  ndprime[dmax] = 0;
  ndprime[0] = 0;
  ndprime[1] = 0;
  for (uint64_t d = dmax-1; d > 1; --d)
    ndprime[d] = ndprime[d+1] + nd[d+1];
  if (verbose && procid == 0) printf(".");

  // Handle degree-1 nodes
  double beta = 1.0;//log((double)nd[1]);
  uint64_t extra1s = (uint64_t)((double)nd[1] * beta) - nd[1];
  num_verts += extra1s;
  ndfill[1] = (uint64_t)((double)nd[1] * beta);
  wd[1] = 0.5 * (double)nd[1];// * beta;
  nd[1] = ndfill[1];
  rdfill[1] = 1.0;
  if (verbose && procid == 0) printf(".");

  // Main loop
  uint64_t g = 0;
  uint64_t nfillblk = 0;
  double intdeg = 0.0;
  for (uint64_t d = 2; d <= dmax; ++d) { 

    //printf("d %d\n", d);
    double wdfilltmp = 0.0;
    if (nfillblk > 0) {
      ndfill[d] = nfillblk < nd[d] ? nfillblk : nd[d];
      nfillblk = nfillblk - ndfill[d];
      wdfilltmp = 0.5 * (double)ndfill[d] * ((double)d - intdeg);
    } 
      
    int64_t ndbulktmp = nd[d] - ndfill[d];
    double wdbulktmp = 0.0;
    if (ndbulktmp > 0) {
      ++g; 
      ig[g] = id[d] + ndfill[d];
      bg[g] = (uint64_t)ceil((double)ndbulktmp / (double)(d+1));
      ng[g] = d + 1;
      if ((bg[g] * (d+1)) > (ndprime[d] + ndbulktmp)) {
        assert(bg[g] == 1);
        ng[g] = ndprime[d] + ndbulktmp;
      } 
      double rho = cbrt(cd[d]);
      pg[g] = rho;
      intdeg = (double)(ng[g] - 1) * rho;
      wdbulktmp = 0.5 * (double)ndbulktmp * ((double)d - intdeg);
      wg[g] = (double)bg[g] * 0.5 * (double)ng[g] * 
                    (double)(ng[g]-1) * log(1.0/(1.0-rho));
      nfillblk = (bg[g] * ng[g]) - ndbulktmp;
    }

    wd[d] = wdbulktmp + wdfilltmp;
    rdfill[d] = wd[d] > 0.0 ? wdfilltmp / wd[d] : 0.0;
  }
  if (verbose && procid == 0) printf(".");

  num_groups = g;
  num_blocks = 0;
#pragma omp parallel for reduction(+:num_blocks)
  for (uint64_t i = 1; i <= num_groups; ++i)
    num_blocks += bg[i];
  if (verbose && procid == 0) printf(".");

  w1 = 0.0;
  w2 = 0.0;
  double* tmp1 = new double[num_groups+1];
  double* tmp2 = new double[dmax+1];

#pragma omp parallel for reduction(+:w1)
  for (uint64_t i = 0; i <= num_groups; ++i) w1 += wg[i];
  parallel_prefixsums(wg, tmp1, num_groups+1);
#pragma omp for
  for (uint64_t i = 1; i <= num_groups; ++i) wg[i] += wg[i-1];
#pragma omp parallel for reduction(+:w2)
  for (uint64_t i = 0; i <= dmax; ++i) w2 += wd[i];
  parallel_prefixsums(wd, tmp2, dmax+1);

  delete [] wd;
  delete [] wg;
  wg = tmp1;
  wd = tmp2;
  if (verbose && procid == 0) printf(".");

  if (debug) {
    printf("%d -- initialize_bter_data() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}

int calculate_parallel_distribution(
  uint64_t dmax, uint64_t* nd, double* wd, 
  uint64_t* ig, uint64_t* bg, uint64_t*& bg_ps, uint64_t* ng, double* pg, 
  uint64_t& start_p1, uint64_t& end_p1, uint64_t& start_p2, uint64_t& end_p2,
  uint64_t& my_gen_edges, uint64_t num_groups, uint64_t num_blocks, double w2)
{
  double elt = omp_get_wtime();
  if (debug) {
    printf("%d -- calculate_parallel_distribution() start ... \n", procid);
  }

  uint64_t* starts_p1 = new uint64_t[nprocs+1];
  uint64_t* starts_p2 = new uint64_t[nprocs+1];
  for (int64_t i = 0; i <= nprocs; ++i) {
    starts_p1[i] = 0;
    starts_p2[i] = 0;
  }

  // determine number of phase 1 edges
  // use that to distribute main loop
  uint64_t* gen_edges = new uint64_t[num_groups+1];
#pragma omp parallel for
  for (uint64_t i = 0; i <= num_groups; ++i)
    gen_edges[i] = 0;

  uint64_t total_edges = 0;
// #ifdef MANYCORE
  bg_ps = new uint64_t[num_groups+1];
  parallel_prefixsums(bg, bg_ps, num_groups + 1);
  // for (uint64_t i = 0; i < num_groups + 1; ++i) 
  //   if (procid == 0) printf("%lu, %lu\n", i, bg_ps[i]);

#pragma omp parallel for reduction(+:total_edges)
  for (uint64_t i = 0; i < num_blocks; ++i) { {
    uint64_t g = binary_search(bg_ps, i, num_groups+1) + 1;
// #else
// #pragma omp parallel for schedule(dynamic,1) reduction(+:total_edges)
//   for (uint64_t g = 1; g <= num_groups; ++g) {
//     for (uint64_t b = 0; b < bg[g]; ++b) {
//#endif
      double p = pg[g];
      if (p == 0.0) continue;

      uint64_t end = (uint64_t)ng[g] * (uint64_t)(ng[g] - 1) / 2;
      gen_edges[g] += (uint64_t)round((double)end * p);
      total_edges += (uint64_t)round((double)end * p);
    }
  }

  uint64_t edges_per_proc = total_edges / (uint64_t)nprocs;
  my_gen_edges += edges_per_proc;
  starts_p1[0] = 1;
  uint64_t cur_edges = 0;
  int64_t cur_proc = 1;
  for (uint64_t g = 1; g <= num_groups; ++g) {
    //if ((cur_edges + gen_edges[g]) > edges_per_proc) {
    cur_edges += gen_edges[g];
    if (cur_edges > edges_per_proc) {
      starts_p1[cur_proc++] = g + 1;
      cur_edges = cur_edges - edges_per_proc;
    }
    assert(cur_proc <= nprocs + 1);
  }
  starts_p1[nprocs] = num_groups;
  start_p1 = starts_p1[procid];
  end_p1 = starts_p1[procid+1];

  delete [] starts_p1;
  delete [] gen_edges;

  gen_edges = new uint64_t[dmax+1];
#pragma omp parallel for
  for (uint64_t i = 0; i <= dmax; ++i)
    gen_edges[i] = 0;

  total_edges = 0;


// #ifdef MANYCORE
#pragma omp parallel for reduction(+:total_edges)
  for (uint64_t k = 2; k < dmax*dmax; ++k) { {
    uint64_t i = k / dmax;
    uint64_t j = k % dmax;
    if (i == 1 || j == 1 || j > i || nd[i] == 0 || nd[j] == 0) continue;
// #else
// #pragma omp parallel for schedule(dynamic,5) reduction(+:total_edges)
//   for (uint64_t i = 2; i <= dmax; ++i) {
//     if (nd[i] == 0) continue;
//     for (uint64_t j = 2; j <= i; ++j) {
//       if (nd[j] == 0) continue;
//#endif
      double d_i = 2.0 * (wd[i] - wd[i-1]) / (double)nd[i];
      double d_j = 2.0 * (wd[j] - wd[j-1]) / (double)nd[j];
      double p = (double)(d_i * d_j) / (w2 * 2.0);
      long end;
      if (i == j)
        end = (uint64_t)nd[i] * (uint64_t)(nd[i] - 1) / 2;
      else 
        end = (uint64_t)nd[i] * (uint64_t)nd[j];

      gen_edges[i] += (uint64_t)round((double)end * p);
      total_edges += (uint64_t)round((double)end * p);
    }
  }

  edges_per_proc = total_edges / (uint64_t)nprocs;
  my_gen_edges += edges_per_proc;
  starts_p2[0] = 2;
  cur_edges = 0;
  cur_proc = 1;
  for (uint64_t i = 0; i <= dmax; ++i) {
    cur_edges += gen_edges[i];
    if (cur_edges > edges_per_proc) {
      starts_p2[cur_proc++] = i + 1;
      cur_edges = cur_edges - edges_per_proc;
    }
    assert(cur_proc <= nprocs + 1);
  }
  starts_p2[nprocs] = dmax + 1;
  start_p2 = starts_p2[procid];
  end_p2 = starts_p2[procid+1];

  delete [] starts_p2;
  delete [] gen_edges;

  if (debug) {
    printf("%d -- calculate_parallel_distribution() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}

int generate_bter_edges(uint64_t dmax, uint64_t* gen_edges, 
  uint64_t* id, uint64_t* nd, double* wd, double* rdfill, uint64_t* ndfill,
  uint64_t* ig, uint64_t* bg, uint64_t* bg_ps, uint64_t* ng, double* pg, 
  uint64_t start_p1, uint64_t end_p1, uint64_t start_p2, uint64_t end_p2,
  uint64_t& phase1_edges, uint64_t& phase2_edges, uint64_t& phase3_edges, 
  uint64_t& num_gen_edges,  uint64_t num_groups, double w2)
{
  double elt = omp_get_wtime();
  double elt1 = omp_get_wtime();
  if (debug) {
    printf("%d -- generate_bter_edges() start ... \n", procid);
    printf("%d -- start_p1 %lu, end_p1 %lu, start_p2 %lu, end_p2 %lu\n",
      procid, start_p1, end_p1, start_p2, end_p2);
  }

  omp_set_nested(1);

#pragma omp parallel reduction(+:phase1_edges) reduction(+:phase2_edges) \
  reduction(+:phase3_edges)
{

  xs1024star_t xs;
  xs1024star_seed((unsigned long)(omp_get_thread_num() + rand()), &xs);
  uint64_t t_edges[THREAD_QUEUE_SIZE];
  uint64_t t_counter = 0;
  uint64_t t_start;

  // for (uint64_t i = 0; i < num_blocks; ++i) { {
  //   uint64_t g = binary_search(bg_ps, i, num_groups+1) + 1;

  // Phase 1
// #ifdef MANYCORE
  uint64_t prev_g = 0;
  uint64_t prev_bg_start = 0;
  uint64_t prev_bg_end = 0;

#pragma omp for schedule(guided)
  for (uint64_t i = bg_ps[start_p1-1]; i < bg_ps[end_p1-1]; ++i) { {
    uint64_t g = binary_search(bg_ps, i, num_groups+1) + 1;
    // if (i < prev_bg_end && i > prev_bg_start)
    //   g = prev_g;
    // else {
    //   prev_g = g;
    //   prev_bg_start = bg_ps[g-1];
    //   prev_bg_end = bg_ps[g];
    // }
    uint64_t b = i - bg_ps[g-1];
// #else
// #pragma omp for schedule(dynamic,1)
//   for (uint64_t g = start_p1; g < end_p1; ++g) {
//     for (uint64_t b = 0; b < bg[g]; ++b) {
//#endif
      double p = pg[g];
      if (p == 0.0) continue;
      uint64_t group_index = ig[g];
      uint64_t block_size = (uint64_t)ng[g];
      uint64_t block_index = group_index + (b * block_size);
      uint64_t end = block_size * (block_size - 1) / 2;
      uint64_t x = 0;
      uint64_t u, v;

      while (x < end) {
        double r = xs1024star_next_real(&xs);
        uint64_t l = (uint64_t)floor( log(r) / log(1.0 - p) );
        x += (l + 1);
        if (x <= end) {
          u = (uint64_t)ceil( (-1.0 + sqrt(1.0 + 8.0*(double)x)) / 2.0 );
          v = x - (u * (u - 1) / 2) - 1;
          uint64_t src = u + block_index;
          uint64_t dst = v + block_index;

          t_edges[t_counter++] = src;
          t_edges[t_counter++] = dst;
          if (t_counter == THREAD_QUEUE_SIZE) {
        #pragma omp atomic capture
            { t_start = num_gen_edges ; num_gen_edges += THREAD_QUEUE_SIZE; }
            
            for (uint64_t l = 0; l < THREAD_QUEUE_SIZE; ++l) {
              gen_edges[t_start+l] = t_edges[l];
            }
            t_counter = 0;
          }
          ++phase1_edges;
        }
      }
    }
  }

  if (debug && omp_get_thread_num() == 0) {
    printf("%d -- phase 1 done: %lf (s)\n", procid, omp_get_wtime() - elt1);
    elt1 = omp_get_wtime();
  }

  // Phase 2
// #ifdef MANYCORE
#pragma omp for schedule(guided)
  for (uint64_t k = start_p2*dmax; k < end_p2*dmax; ++k) { {
    uint64_t i = k / dmax;
    uint64_t j = k % dmax;
    if (i == 1 || j == 1 || j > i || nd[i] == 0 || nd[j] == 0) continue;
// #else
// #pragma omp for schedule(dynamic,1)
//   for (uint64_t i = start_p2; i < end_p2; ++i) {
//     if (nd[i] == 0) continue;
//     for (uint64_t j = 2; j <= i; ++j) {
//       if (nd[j] == 0) continue;
//#endif
      double d_i = 2.0 * (wd[i] - wd[i-1]) / (double)nd[i];
      double d_j = 2.0 * (wd[j] - wd[j-1]) / (double)nd[j];
      double p = (double)(d_i * d_j) / (w2 * 2.0);
      uint64_t degree_index_i = id[i];
      uint64_t degree_index_j = id[j];
      uint64_t end, u, v;

      if (i == j)
        end = (uint64_t)nd[i] * (uint64_t)(nd[i] - 1) / 2;
      else 
        end = (uint64_t)nd[i] * (uint64_t)nd[j];

      uint64_t x = 0;
      while (x < end && x >= 0) {
        double r = xs1024star_next_real(&xs);
        uint64_t l = (uint64_t)floor( log(r) / log(1.0 - p) );
        x += (l + 1);
        if (x <= end && x >= 0) {
          if (i == j) {
            u = (uint64_t)ceil( (-1.0 + sqrt(1.0 + 8.0*(double)x)) / 2.0 );
            v = x - (u * (u - 1) / 2) - 1;
          } else {
            u = (uint64_t)floor( ((double)x - 1.0) / (double)nd[j] );
            v = (x - 1) % nd[j];
          }
          
          uint64_t src = u + degree_index_i;
          uint64_t dst = v + degree_index_j;

          // check for block association
          if ((uint64_t)abs((int64_t)src - (int64_t)dst) > (min(i, j))) {
            t_edges[t_counter++] = src;
            t_edges[t_counter++] = dst;
            if (t_counter == THREAD_QUEUE_SIZE) {
          #pragma omp atomic capture
              { t_start = num_gen_edges ; num_gen_edges += THREAD_QUEUE_SIZE; }
              
              for (uint64_t l = 0; l < THREAD_QUEUE_SIZE; ++l) {
                gen_edges[t_start+l] = t_edges[l];
              }
              t_counter = 0;
            }
            ++phase2_edges;
          }
        }
      }
    }
  }
#pragma omp atomic capture
  { t_start = num_gen_edges ; num_gen_edges += t_counter; }
  
  for (uint64_t l = 0; l < t_counter; ++l) {
    gen_edges[t_start+l] = t_edges[l];;
  }

  // Phase 3 -- e.g., 1-degree vertices
  uint64_t my_start = (uint64_t)procid * nd[1] / nprocs;
  uint64_t my_end = (uint64_t)(procid+1) * nd[1] / nprocs;
  if (procid == nprocs - 1) my_end = nd[1];
#pragma omp for 
  for (uint64_t i = my_start; i < my_end; ++i) {
    uint64_t src = i + id[1];
    uint64_t dst = 0;
    double r = 0.0;
    uint64_t d = 0;
    do {
      r = xs1024star_next_real(&xs) * w2;
      d = binary_search(wd, r, dmax+1) + 1;
    } while (d == 1);
    r = xs1024star_next_real(&xs);
    if (r < rdfill[d]) {
      dst = (int)floor(r * (double)ndfill[d]) + id[d];
    } else {
      dst = (int)floor(r * (double)(nd[d] - ndfill[d])) + (id[d] + ndfill[d]);
    }

    t_edges[t_counter++] = src;
    t_edges[t_counter++] = dst;
    if (t_counter == THREAD_QUEUE_SIZE) {
  #pragma omp atomic capture
      { t_start = num_gen_edges ; num_gen_edges += THREAD_QUEUE_SIZE; }
      
      for (uint64_t l = 0; l < THREAD_QUEUE_SIZE; ++l) {
        gen_edges[t_start+l] = t_edges[l];
      }
      t_counter = 0;
    }
    ++phase3_edges;
  }
#pragma omp atomic capture
  { t_start = num_gen_edges ; num_gen_edges += t_counter; }
  
  for (uint64_t l = 0; l < t_counter; ++l) {
    gen_edges[t_start+l] = t_edges[l];;
  }
} // end parallel

  if (debug) {
    printf("%d -- phase 2 done: %lf (s)\n", procid, omp_get_wtime() - elt1);
    printf("%d -- phase 1: %lu, phase 2: %lu, phase 3: %lu\n", 
      procid, phase1_edges, phase2_edges, phase3_edges);
    printf("%d -- generate_bter_edges() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}


int exchange_edges(uint64_t* nd, uint64_t dmax, uint64_t* id,
  uint64_t*& gen_edges, uint64_t& num_edges, uint64_t& num_gen_edges, 
  uint64_t num_verts, uint64_t& local_offset)
{
  double elt = omp_get_wtime();
  if (debug) { 
    printf("%d -- exchange_edges() start ... \n", procid); 
  }

  uint64_t* starts = new uint64_t[nprocs+1];
  starts[0] = 0;
  uint64_t cur_edges = 0;
  int32_t cur_proc = 1;
  uint64_t m_per_rank = (num_gen_edges*2) / (uint64_t)nprocs;
  for (uint64_t i = 2; i <= dmax; ++i) {
    cur_edges += nd[i]*i;
    if (cur_edges > m_per_rank) {
      starts[cur_proc++] = id[i + 1];
      cur_edges = cur_edges - m_per_rank;
    }
    if (cur_proc == nprocs) break;
    //assert(cur_proc <= nprocs);
  }
  if (cur_proc == nprocs - 1)
    starts[nprocs-1] = nd[1];
  // cur_edges += nd[1];
  // if (cur_edges > m_per_rank) {
  //   starts[cur_proc++] = id[1];
  //   cur_edges = cur_edges - m_per_rank;
  // }
  //assert(cur_proc <= nprocs+1);
  starts[nprocs] = num_verts;
  local_offset = starts[procid];

  uint64_t* all_sendcounts = new uint64_t[nprocs];
  uint64_t* all_recvcounts = new uint64_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i) {
    all_sendcounts[i] = 0;
    all_recvcounts[i] = 0;
  }

#pragma omp parallel
{
  uint64_t* t_sendcounts = new uint64_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i)
    t_sendcounts[i] = 0;

#pragma omp for
  for (uint64_t i = 0; i < num_edges*2; i+=2) {
    uint64_t src = gen_edges[i];
    uint64_t dst = gen_edges[i+1];    
    int32_t src_task = binary_search(starts, src, (uint64_t)nprocs);
    int32_t dst_task = binary_search(starts, dst, (uint64_t)nprocs);
    if (src_task != dst_task) {
      t_sendcounts[src_task] += 2;
      t_sendcounts[dst_task] += 2;
    } else {
      t_sendcounts[src_task] += 2;
    }
  }

  for (int32_t i = 0; i < nprocs; ++i)
#pragma omp atomic
    all_sendcounts[i] += t_sendcounts[i];

  delete [] t_sendcounts;
} // end parallel

  MPI_Alltoall(all_sendcounts, 1, MPI_UINT64_T, 
               all_recvcounts, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += all_recvcounts[i];
    total_send += all_sendcounts[i];
  }
  delete [] all_sendcounts;
  delete [] all_recvcounts;

  uint64_t* recvbuf = new uint64_t[total_recv];
  if (recvbuf == NULL) { 
    fprintf(stderr, 
      "Error: %d -- exchange_out_edges(), unable to allocate buffer\n", 
      procid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }  

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE/2) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("%d -- exchange_edges() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);


  int32_t* sendcounts = new int32_t[nprocs];
  int32_t* recvcounts = new int32_t[nprocs];
  int32_t* sdispls = new int32_t[nprocs];
  int32_t* sdispls_cpy = new int32_t[nprocs];
  int32_t* rdispls = new int32_t[nprocs];

  uint64_t sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (num_edges * c) / num_comms;
    uint64_t send_end = (num_edges * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = num_edges;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      sendcounts[i] = 0;
      recvcounts[i] = 0;
    }
#pragma omp parallel
{
  int32_t* t_sendcounts = new int32_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i)
    t_sendcounts[i] = 0;

  #pragma omp for
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t src = gen_edges[2*i];
      uint64_t dst = gen_edges[2*i+1];
      int32_t src_task = binary_search(starts, src, (uint64_t)nprocs);
      int32_t dst_task = binary_search(starts, dst, (uint64_t)nprocs);
      if (src_task != dst_task) {
        t_sendcounts[src_task] += 2;
        t_sendcounts[dst_task] += 2;
      } else {
        t_sendcounts[src_task] += 2;
      }
    }

    for (int32_t i = 0; i < nprocs; ++i)
  #pragma omp atomic
      sendcounts[i] += t_sendcounts[i];

    delete [] t_sendcounts;
} // end parallel

    MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
                 recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    sdispls[0] = 0;
    sdispls_cpy[0] = 0;
    rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
      sdispls_cpy[i] = sdispls[i];
    }

    int32_t cur_send = sdispls[nprocs-1] + sendcounts[nprocs-1];
    int32_t cur_recv = rdispls[nprocs-1] + recvcounts[nprocs-1];
    uint64_t* sendbuf = new uint64_t[cur_send];
    if (sendbuf == NULL)
    { 
      fprintf(stderr, 
        "Error: %d -- exchange_out_edges(), unable to allocate comm buffers",
         procid);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t src = gen_edges[2*i];
      uint64_t dst = gen_edges[2*i+1];
      int32_t src_task = binary_search(starts, src, (uint64_t)nprocs);
      int32_t dst_task = binary_search(starts, dst, (uint64_t)nprocs);

      if (src_task != dst_task) {
        sendbuf[sdispls_cpy[src_task]++] = src; 
        sendbuf[sdispls_cpy[src_task]++] = dst;
        sendbuf[sdispls_cpy[dst_task]++] = src; 
        sendbuf[sdispls_cpy[dst_task]++] = dst;
      } else {
        sendbuf[sdispls_cpy[src_task]++] = src; 
        sendbuf[sdispls_cpy[src_task]++] = dst;
      }
    }

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_UINT64_T, 
                  recvbuf+sum_recv, recvcounts, rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv += (uint64_t)cur_recv;
    delete [] sendbuf;
  }

  delete [] starts;
  delete [] gen_edges;
  gen_edges = recvbuf;
  num_edges = total_recv / 2;

  if (debug) {
    printf("%d -- sent %lu, recv %lu, local_edges %lu\n", 
      procid, total_send, total_recv, num_edges);
    printf("%d -- exchange_out_edges() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}

int assign_ground_truth(uint64_t* bg, uint64_t* ng, uint64_t& num_groups,
   uint64_t num_verts, int64_t*& ground_truth)
{
  double elt = omp_get_wtime();
  if (debug) { 
    printf("%d -- assign_ground_truth() start ... \n", procid); 
  }

  uint64_t id_start = (uint64_t)procid * (num_verts / (uint64_t)nprocs + 1);
  uint64_t id_end = (uint64_t)(procid + 1) * (num_verts / (uint64_t)nprocs + 1);
  if (procid == nprocs-1)
    id_end = num_verts;

  uint64_t num_local_verts = id_end - id_start;

  if (debug) {
    printf("%d -- id_start: %li, id_end: %lu, num_local_verts: %lu\n", 
      procid, id_start, id_end, num_local_verts);
  }

  ground_truth = new int64_t[num_local_verts];
#pragma omp parallel for
  for (uint64_t i = 0; i < num_local_verts; ++i)
    ground_truth[i] = -1;

  uint64_t counter = 0;
  uint64_t local_counter = 0;
  uint64_t comm = 0;
  for (uint64_t i = 1; i <= num_groups; ++i) {
    for (uint64_t j = 0; j < bg[i]; ++j) {
      for (uint64_t k = 0; k < ng[i]; ++k) {
        if (counter >= id_start)
          ground_truth[local_counter++] = comm;
        assert(local_counter <= num_local_verts);
        ++counter;
        if (counter >= id_end)
          goto done;
      }
      ++comm;
    }
  }

  for (uint64_t i = counter; i < num_verts; ++i) {
    if (counter >= id_start)
      ground_truth[local_counter++] = comm++;
    assert(local_counter <= num_local_verts);
    ++counter;
    if (counter >= id_end)
      break;
  }
done:

  if (debug) {
    printf("%d -- assign_ground_truth() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }
  return 0;
}


int assign_ground_truth_no_zeros(uint64_t* nd, uint64_t dmax, 
  uint64_t* id, uint64_t* bg, uint64_t* ng, uint64_t& num_groups,
  uint64_t& num_gen_edges, uint64_t num_verts, int64_t*& ground_truth)
{
  double elt = omp_get_wtime();
  if (debug) { 
    printf("%d -- assign_ground_truth() start ... \n", procid); 
  }

  uint64_t* starts = new uint64_t[nprocs+1];
  starts[0] = 0;
  uint64_t cur_edges = 0;
  int32_t cur_proc = 1;
  uint64_t m_per_rank = (num_gen_edges*2) / (uint64_t)nprocs;
  for (uint64_t i = 2; i <= dmax; ++i) {
    cur_edges += nd[i]*i;
    if (cur_edges > m_per_rank) {
      starts[cur_proc++] = id[i + 1];
      cur_edges = cur_edges - m_per_rank;
    }
    if (cur_proc == nprocs) break;
    //assert(cur_proc <= nprocs);
  }
  if (cur_proc == nprocs - 1)
    starts[nprocs-1] = nd[1];
  starts[nprocs] = num_verts;

  uint64_t id_start = starts[procid];
  uint64_t id_end = starts[procid+1];
  uint64_t num_local_verts = id_end - id_start;
  delete [] starts;

  if (debug) {
    printf("%d -- id_start: %li, id_end: %lu, num_local_verts: %lu\n", 
      procid, id_start, id_end, num_local_verts);
  }

  ground_truth = new int64_t[num_local_verts];
#pragma omp parallel for
  for (uint64_t i = 0; i < num_local_verts; ++i)
    ground_truth[i] = -1;

  uint64_t counter = 0;
  uint64_t local_counter = 0;
  uint64_t comm = 0;
  for (uint64_t i = 1; i <= num_groups; ++i) {
    for (uint64_t j = 0; j < bg[i]; ++j) {
      for (uint64_t k = 0; k < ng[i]; ++k) {
        if (counter >= id_start)
          ground_truth[local_counter++] = comm;
        assert(local_counter <= num_local_verts);
        ++counter;
        if (counter >= id_end)
          goto done;
      }
      ++comm;
    }
  }

  for (uint64_t i = counter; i < num_verts; ++i) {
    if (counter >= id_start)
      ground_truth[local_counter++] = comm++;
    assert(local_counter <= num_local_verts);
    ++counter;
    if (counter >= id_end)
      break;
  }
done:

  if (debug) {
    printf("%d -- assign_ground_truth() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }
  return 0;
}


// To remove zero degree vertices
// --Initially have local edges based on starts/offsets
// --Get new local count based on which vertices have edges
// --Exchange new local counts for new starts/offsets
// --Remap locally-owned vertices using new starts
// --Exchange cross-boundary edges
// --Remap received edges
// --Final exchange of boundary edges
int remove_zero_degrees(uint64_t* nd, uint64_t dmax, uint64_t* id,
  uint64_t*& gen_edges, uint64_t& num_edges, uint64_t& num_gen_edges, 
  uint64_t& num_verts, int64_t*& ground_truth,
  uint64_t& num_local_verts, uint64_t& num_local_edges)
{
  double elt = omp_get_wtime();
  if (debug) { 
    printf("%d -- remove_zero_degrees() start ... \n", procid); 
  }

  uint64_t* starts = new uint64_t[nprocs+1];
  starts[0] = 0;
  uint64_t cur_edges = 0;
  int32_t cur_proc = 1;
  uint64_t m_per_rank = (num_gen_edges*2) / (uint64_t)nprocs;
  for (uint64_t i = 2; i <= dmax; ++i) {
    cur_edges += nd[i]*i;
    if (cur_edges > m_per_rank) {
      starts[cur_proc++] = id[i + 1];
      cur_edges = cur_edges - m_per_rank;
    }
    if (cur_proc == nprocs) break;
    //assert(cur_proc <= nprocs);
  }
  if (cur_proc == nprocs - 1)
    starts[nprocs-1] = nd[1];
  starts[nprocs] = num_verts;

  // for (int32_t i = 0; i <= nprocs; ++i)
  //   printf("%d %lu\n", i, starts[i]);

  uint64_t id_start = starts[procid];
  uint64_t id_end = starts[procid+1];
  uint64_t local_verts = id_end - id_start;

// --Get new local count based on which vertices have edges
  bool* has_edge = new bool[local_verts];
#pragma omp parallel for
  for (uint64_t i = 0; i < local_verts; ++i)
    has_edge[i] = false;

#pragma omp parallel for
  for (uint64_t i = 0; i < num_edges*2; ++i) {
    uint64_t vert = gen_edges[i];
    if (vert >= id_start && vert < id_end)
      has_edge[vert - id_start] = true;
  }

  uint64_t new_local_verts = 0;
#pragma omp parallel for reduction(+:new_local_verts)
  for (uint64_t i = 0; i < local_verts; ++i)
    if (has_edge[i])
      ++new_local_verts;

// --Exchange new local counts for new starts/offsets
  uint64_t* new_starts = new uint64_t[nprocs+1];
  for (int32_t i = 0; i <= nprocs; ++i)
    new_starts[i] = 0;

  MPI_Allgather(&new_local_verts, 1, MPI_UINT64_T,
    &new_starts[1], 1, MPI_UINT64_T, MPI_COMM_WORLD);

  for (int32_t i = 1; i <= nprocs; ++i)
    new_starts[i] = new_starts[i] + new_starts[i-1];
  num_verts = new_starts[nprocs];

// --Remap locally-owned vertices using new starts
  int64_t* map = new int64_t[local_verts];
#pragma omp parallel for
  for (uint64_t i = 0; i < local_verts; ++i)
    map[i] = -1;

  uint64_t new_id_start = new_starts[procid];
  for (uint64_t i = 0; i < local_verts; ++i)
    if (has_edge[i])
      map[i] = new_id_start++;

  assert(new_id_start == new_starts[procid+1]);  
  uint64_t new_id_end = new_id_start;
  new_id_start = new_starts[procid];

  num_local_verts = new_id_end - new_id_start;

  if (debug) {
    printf("%d -- Num zero-degree verts: %lu\n", 
      procid, local_verts - num_local_verts);
  }

// --update ground truths based on new mappings
  int64_t* new_ground_truth = new int64_t[num_local_verts];

#pragma omp parallel for
  for (uint64_t i = 0; i < local_verts; ++i)
    if (map[i] >= 0)
      new_ground_truth[map[i]-new_id_start] = ground_truth[i];

  delete [] ground_truth;
  ground_truth = new_ground_truth;

// --Exchange cross-boundary edges
  uint64_t num_intratask_edges = 0;
  uint64_t num_boundary_edges = 0;
  num_local_edges = 0;

#pragma omp parallel for reduction(+:num_intratask_edges) \
                          reduction(+:num_boundary_edges)
  for (uint64_t i = 0; i < num_edges*2; i+=2) {
    uint64_t src = gen_edges[i];
    uint64_t dst = gen_edges[i+1];
    if (src >= id_start && src < id_end && 
        dst >= id_start && dst < id_end)
      ++num_intratask_edges;
    else
      ++num_boundary_edges;
  } 

  num_local_edges = num_intratask_edges + num_boundary_edges;

  if (debug) {
    printf("%d -- local_edges %lu, intratask_edges %lu, boundary_edges %lu\n",
      procid, num_local_edges, num_intratask_edges, num_boundary_edges);
  }

  uint64_t* new_gen_edges = new uint64_t[num_edges*2];
  uint64_t local_offset = 0;
  uint64_t boundary_offset = num_intratask_edges*2;

#pragma omp parallel 
{  
  uint64_t* t_local = new uint64_t[THREAD_QUEUE_SIZE/2];
  uint64_t* t_boundary = new uint64_t[THREAD_QUEUE_SIZE/2];
  uint64_t t_offset_local = 0;
  uint64_t t_offset_boundary = 0;
  uint64_t t_start = 0;

#pragma omp for
  for (uint64_t i = 0; i < num_edges*2; i+=2) {
    uint64_t src = gen_edges[i];
    uint64_t dst = gen_edges[i+1];
    if (src >= id_start && src < id_end && 
        dst >= id_start && dst < id_end) {
      t_local[t_offset_local++] = src;
      t_local[t_offset_local++] = dst;
      if (t_offset_local == THREAD_QUEUE_SIZE/2) {
    #pragma omp atomic capture
        { t_start = local_offset ; local_offset += THREAD_QUEUE_SIZE/2; }
        
        for (uint64_t l = 0; l < THREAD_QUEUE_SIZE/2; ++l) {
          new_gen_edges[t_start+l] = t_local[l];
        }
        t_offset_local = 0;
      }
    } else {
      t_boundary[t_offset_boundary++] = src;
      t_boundary[t_offset_boundary++] = dst;
      if (t_offset_boundary == THREAD_QUEUE_SIZE/2) {
    #pragma omp atomic capture
        { t_start = boundary_offset ; boundary_offset += THREAD_QUEUE_SIZE/2; }
        
        for (uint64_t l = 0; l < THREAD_QUEUE_SIZE/2; ++l) {
          new_gen_edges[t_start+l] = t_boundary[l];
        }
        t_offset_boundary = 0;
      }
    }
  }
#pragma omp atomic capture
  { t_start = local_offset ; local_offset += t_offset_local; }
  
  for (uint64_t l = 0; l < t_offset_local; ++l)
    new_gen_edges[t_start+l] = t_local[l];

#pragma omp atomic capture
  { t_start = boundary_offset ; boundary_offset += t_offset_boundary; }

  for (uint64_t l = 0; l < t_offset_boundary; ++l)
    new_gen_edges[t_start+l] = t_boundary[l];

  delete [] t_local;
  delete [] t_boundary;
}
  delete [] gen_edges;
  gen_edges = new_gen_edges;

  // now have new_gen_edges with [(local edges)(boundary edges)]
  // send boundary edges away to be relabeled
  if (debug) {
    assert(local_offset == num_intratask_edges*2);
    assert(boundary_offset == num_edges*2);
    printf("%d -- Local offset: %lu, Boundary offset: %lu\n", 
      procid, local_offset, boundary_offset);
  }

  local_offset = 0;
  boundary_offset = num_intratask_edges*2;
  uint64_t* boundary_edges = &gen_edges[boundary_offset];

  int32_t* sendcounts = new int32_t[nprocs];
  int32_t* recvcounts = new int32_t[nprocs];
  int32_t* sdispls = new int32_t[nprocs];
  int32_t* sdispls_cpy = new int32_t[nprocs];
  int32_t* rdispls = new int32_t[nprocs];

  uint64_t* all_sendcounts = new uint64_t[nprocs];
  uint64_t* all_recvcounts = new uint64_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i) {
    all_sendcounts[i] = 0;
    all_recvcounts[i] = 0;
  }

  // compute send counts
#pragma omp parallel
{
  uint64_t* t_sendcounts = new uint64_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i)
    t_sendcounts[i] = 0;

#pragma omp for
  for (uint64_t i = boundary_offset; i < num_edges*2; i+=2) {
    uint64_t src = new_gen_edges[i];
    uint64_t dst = new_gen_edges[i+1];
    if (src < id_start || src >= id_end) {
      int32_t src_task = binary_search(starts, src, (uint64_t)nprocs);
      t_sendcounts[src_task] += 2;
    }
    else if (dst < id_start || dst >= id_end) {
      int32_t dst_task = binary_search(starts, dst, (uint64_t)nprocs);
      t_sendcounts[dst_task] += 2;
    }
    else
      printf("Error\n");
  }

  for (int32_t i = 0; i < nprocs; ++i)
#pragma omp atomic
    all_sendcounts[i] += t_sendcounts[i];

  delete [] t_sendcounts;
} // end parallel

  MPI_Alltoall(all_sendcounts, 1, MPI_UINT64_T, 
               all_recvcounts, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += all_recvcounts[i];
    total_send += all_sendcounts[i];
  }
  delete [] all_sendcounts;
  delete [] all_recvcounts;

  if (debug) {
    printf("%d -- Total Recv: %lu, Total Send: %lu\n", 
      procid, total_recv, total_send);
  }

  uint64_t* recvbuf = new uint64_t[total_recv];
  if (recvbuf == NULL) { 
    fprintf(stderr, 
      "Error: %d -- remove_zero_degrees(), unable to allocate buffer\n", 
      procid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }  

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE/2) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  
  uint64_t sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (num_boundary_edges * c) / num_comms;
    uint64_t send_end = (num_boundary_edges * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = num_boundary_edges;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      sendcounts[i] = 0;
      recvcounts[i] = 0;
    }
#pragma omp parallel
{
  int32_t* t_sendcounts = new int32_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i)
    t_sendcounts[i] = 0;

  #pragma omp for
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t src = boundary_edges[i*2];
      uint64_t dst = boundary_edges[i*2+1];
      int32_t send_task = -1;
      if (src < id_start || src >= id_end) {
        send_task = binary_search(starts, src, (uint64_t)nprocs);
      }
      else if (dst < id_start || dst >= id_end) {
        send_task = binary_search(starts, dst, (uint64_t)nprocs);
      }
      t_sendcounts[send_task] += 2;
    }

    for (int32_t i = 0; i < nprocs; ++i)
  #pragma omp atomic
      sendcounts[i] += t_sendcounts[i];

    delete [] t_sendcounts;
} // end parallel

    MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
                 recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    sdispls[0] = 0;
    sdispls_cpy[0] = 0;
    rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
      sdispls_cpy[i] = sdispls[i];
    }

    int32_t cur_send = sdispls[nprocs-1] + sendcounts[nprocs-1];
    int32_t cur_recv = rdispls[nprocs-1] + recvcounts[nprocs-1];
    uint64_t* sendbuf = new uint64_t[cur_send];
    if (sendbuf == NULL)
    { 
      fprintf(stderr, 
        "Error: %d -- remove_zero_degrees(), unable to allocate comm buffers",
         procid);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t src = boundary_edges[i*2];
      uint64_t dst = boundary_edges[i*2+1];
      int32_t send_task = -1;
      if (src < id_start || src >= id_end) {
        send_task = binary_search(starts, src, (uint64_t)nprocs);
      }
      else if (dst < id_start || dst >= id_end) {
        send_task = binary_search(starts, dst, (uint64_t)nprocs);
      }
      sendbuf[sdispls_cpy[send_task]++] = src; 
      sendbuf[sdispls_cpy[send_task]++] = dst;
    }

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_UINT64_T, 
                  recvbuf+sum_recv, recvcounts, rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv += (uint64_t)cur_recv;
    delete [] sendbuf;
  }

  if (debug) {
    printf("%d -- Sum Recv: %lu\n", procid, sum_recv);
  }

// --remap and resend edges
  bool* updated = new bool[sum_recv];
  sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (num_boundary_edges * c) / num_comms;
    uint64_t send_end = (num_boundary_edges * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = num_boundary_edges;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      sendcounts[i] = 0;
      recvcounts[i] = 0;
    }
#pragma omp parallel
{
  int32_t* t_sendcounts = new int32_t[nprocs];
  for (int32_t i = 0; i < nprocs; ++i)
    t_sendcounts[i] = 0;

  #pragma omp for
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t src = recvbuf[i*2];
      uint64_t dst = recvbuf[i*2+1];
      int32_t send_task = -1;
      if (src < id_start || src >= id_end) {
        send_task = binary_search(starts, src, (uint64_t)nprocs);
      }
      else if (dst < id_start || dst >= id_end) {
        send_task = binary_search(starts, dst, (uint64_t)nprocs);
      }
      t_sendcounts[send_task] += 2;
    }

    for (int32_t i = 0; i < nprocs; ++i)
  #pragma omp atomic
      sendcounts[i] += t_sendcounts[i];

    delete [] t_sendcounts;
} // end parallel

    MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
                 recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    sdispls[0] = 0;
    sdispls_cpy[0] = 0;
    rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
      sdispls_cpy[i] = sdispls[i];
    }

    int32_t cur_send = sdispls[nprocs-1] + sendcounts[nprocs-1];
    int32_t cur_recv = rdispls[nprocs-1] + recvcounts[nprocs-1];
    uint64_t* sendbuf = new uint64_t[cur_send];
    bool* updated_send = new bool[cur_send];
    if (sendbuf == NULL)
    { 
      fprintf(stderr, 
        "Error: %d -- remove_zero_degrees(), unable to allocate comm buffers",
         procid);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t src = recvbuf[i*2];
      uint64_t dst = recvbuf[i*2+1];
      int32_t send_task = -1;
      if (src < id_start || src >= id_end) {
        send_task = binary_search(starts, src, (uint64_t)nprocs);
        dst = map[dst-id_start];
        updated_send[sdispls_cpy[send_task]] = false;
        updated_send[sdispls_cpy[send_task]+1] = true;
      }
      else if (dst < id_start || dst >= id_end) {
        send_task = binary_search(starts, dst, (uint64_t)nprocs);
        src = map[src-id_start];
        updated_send[sdispls_cpy[send_task]] = true;
        updated_send[sdispls_cpy[send_task]+1] = false;
      }
      sendbuf[sdispls_cpy[send_task]++] = src;
      sendbuf[sdispls_cpy[send_task]++] = dst;
    }

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_UINT64_T, 
                  boundary_edges+sum_recv, recvcounts, rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    MPI_Alltoallv(updated_send, sendcounts, sdispls, MPI::BOOL, 
                  updated+sum_recv, recvcounts, rdispls,
                  MPI::BOOL, MPI_COMM_WORLD);
    sum_recv += (uint64_t)cur_recv;
    delete [] sendbuf;
    delete [] updated_send;
  }

  if (debug) {
    printf("%d -- Sum Recv: %lu\n", procid, sum_recv);
  }

// --finalize all local mapping
#pragma omp parallel for
  for (uint64_t i = 0; i < num_intratask_edges*2; ++i) {
    gen_edges[i] = map[gen_edges[i]-id_start];
    
    assert(gen_edges[i] >= new_id_start);
    assert(gen_edges[i] < new_id_end);
  }

// --finalize non-local mapping
// --make edges unique system-wide by only retaining edges where src is local
  uint64_t cur_counter = 0;
//#pragma omp parallel for
  for (uint64_t i = 0; i < num_boundary_edges*2; i+=2) {
    //uint64_t tmp = 0;
    if (!updated[i]) {      
      boundary_edges[cur_counter++] = map[boundary_edges[i]-id_start];
      boundary_edges[cur_counter++] = boundary_edges[i+1];
      //assert(boundary_edges[i] >= new_id_start);
      //assert(boundary_edges[i] < new_id_end);
    } /*else if (!updated[i+1]) {
      tmp = boundary_edges[i];
      boundary_edges[i] = map[boundary_edges[i+1]-id_start];
      boundary_edges[i+1] = tmp;
    } else {
      printf("Error\n");
    } */
  }  

  if (debug) {
    printf("%d -- new num_boundary_edges: %lu\n", procid, cur_counter / 2);
  }

  num_local_edges = num_intratask_edges + cur_counter / 2;

  delete [] map;
  delete [] recvbuf;
  delete [] updated;
  delete [] starts;
  delete [] sendcounts;
  delete [] recvcounts;
  delete [] sdispls;
  delete [] sdispls_cpy;
  delete [] rdispls;

  if (debug) {
    printf("%d -- remove_zero_degrees() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}

int generate_bter_MPI(char* nd_filename, char* cd_filename, 
  uint64_t*& gen_edges, int64_t*& ground_truth,
  uint64_t& num_local_edges, uint64_t& num_local_verts, uint64_t& local_offset,
  bool remove_zeros)
{
  double total_elt = omp_get_wtime();
  double elt = omp_get_wtime();  
  if (debug) {
    printf("%d -- generate_bter_MPI() start ... \n", procid);
  }

  gen_edges = NULL;
  uint64_t* nd = NULL;
  double* cd = NULL;
  uint64_t dmax = 0;
  uint64_t num_verts = 0;
  uint64_t num_edges = 0;
 
  read_nd_cd(nd_filename, cd_filename, nd, cd, num_verts, num_edges, dmax);

  if (verbose && procid == 0) {
    printf("Beginning BTER preprocessing\n");
  }

  uint64_t* id = new uint64_t[dmax+1]; 
  double* wd = new double[dmax+1];
  double* rdfill = new double[dmax+1];
  uint64_t* ndfill = new uint64_t[dmax+1]; 
  double* wg = new double[dmax+1];
  uint64_t* ig = new uint64_t[dmax+1];
  uint64_t* bg = new uint64_t[dmax+1];
  uint64_t* bg_ps = NULL;
  uint64_t* ng = new uint64_t[dmax+1];
  double* pg = new double[dmax+1]; 
  uint64_t* ndprime = new uint64_t[dmax+1];
  uint64_t num_groups = 0;
  uint64_t num_blocks = 0;  
  double w1 = 0.0;
  double w2 = 0.0;

  initialize_bter_data(nd, cd,
    dmax, num_verts,
    id, wd, rdfill, ndfill, 
    wg, ig, bg, ng, pg, 
    ndprime, num_groups, num_blocks,
    w1, w2);

  if (verbose && procid == 0) {
    printf(" Done: %lf\n", omp_get_wtime() - elt);
    printf("Number of groups: %lu\n", num_groups);
    printf("Number of blocks: %lu\n", num_blocks);
    printf("Phase 1 total weight: %.0lf\n", w1);
    printf("Phase 2 total weight: %.0lf\n", w2);
    printf("Beginning BTER edge generation \n");
  }

  uint64_t my_gen_edges = 0;
  uint64_t start_p1, end_p1, start_p2, end_p2;

  calculate_parallel_distribution(
    dmax, nd, wd,
    ig, bg, bg_ps, ng, pg, 
    start_p1, end_p1, start_p2, end_p2,
    my_gen_edges, num_groups, num_blocks, w2);

  uint64_t alloc_size = (my_gen_edges + nd[1] / (uint64_t)nprocs)*2;
  alloc_size = (uint64_t)((double)alloc_size*1.05);
  gen_edges = new uint64_t[alloc_size];
  if (debug) {
    printf("%d -- gen_edges allocation size: %lu\n", procid, alloc_size);
  }

  uint64_t phase1_edges = 0;
  uint64_t phase2_edges = 0;
  uint64_t phase3_edges = 0;
  uint64_t num_gen_edges = 0;

  generate_bter_edges(dmax, gen_edges,
    id, nd, wd, rdfill, ndfill,
    ig, bg, bg_ps, ng, pg,
    start_p1, end_p1, start_p2, end_p2,
    phase1_edges, phase2_edges, phase3_edges,
    num_gen_edges, num_groups, w2);

  num_gen_edges /= 2;
  num_edges = num_gen_edges;

  delete [] cd;
  delete [] wd;
  delete [] rdfill;
  delete [] ndfill;
  delete [] ig;
  delete [] wg;
// #ifdef MANYCORE
  delete [] bg_ps;
// #endif
  delete [] ndprime;

  MPI_Allreduce(MPI_IN_PLACE, &num_gen_edges, 1, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &phase1_edges, 1, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &phase2_edges, 1, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &phase3_edges, 1, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (verbose && procid == 0) {
    printf(" Done: %lf\n", omp_get_wtime() - elt);
    printf("Generated %lu total edges\n", num_gen_edges);
    printf("Generated %lu edges for phase 1\n", phase1_edges);
    printf("Generated %lu edges for phase 2\n", phase2_edges);
    printf("Generated %lu edges for phase 3\n", phase3_edges);
    printf("Number of edges created by BTER: %li\n", phase1_edges+phase2_edges);
  }

  if (remove_zeros) {
    if (verbose && procid == 0) {
      printf("Beginning edge transfer ... \n");
      elt = omp_get_wtime();
    }

    exchange_edges(nd, dmax, id,
      gen_edges, num_edges, num_gen_edges, num_verts, local_offset);

    if (verbose && procid == 0) {
      printf(" Done: %lf\n", omp_get_wtime() - elt);
      printf("Assigning ground truth ... ");
      elt = omp_get_wtime();
    }

    assign_ground_truth_no_zeros(nd, dmax, 
      id, bg, ng, num_groups,
      num_gen_edges, num_verts, ground_truth);

    if (verbose && procid == 0) {
      printf(" Done: %lf\n", omp_get_wtime() - elt);
      printf("Beginning zero-degree vertex removal ... ");
      elt = omp_get_wtime();
    }

    remove_zero_degrees(nd, dmax, id,
      gen_edges, num_edges, num_gen_edges, 
      num_verts, ground_truth,
      num_local_verts, num_local_edges);

    if (verbose && procid == 0) {
      printf("New num_verts: %lu\n", num_verts);
      printf(" Done: %lf\n", omp_get_wtime() - elt);
    }

  } else {
    if (verbose && procid == 0) {
      printf("Assigning ground truth ... ");
      elt = omp_get_wtime();
    }

    num_local_edges = num_edges;
    local_offset = (uint64_t)procid * (num_verts / (uint64_t)nprocs + 1);
    
    assign_ground_truth(bg, ng, num_groups,
      num_verts, ground_truth);

    if (verbose && procid == 0) {
      printf(" Done: %lf\n", omp_get_wtime() - elt);
    }
  }

  delete [] ng;
  delete [] bg;
  delete [] id;
  free(nd);

  if (verbose && procid == 0)
    printf("Total BTER generation time: %lf\n", omp_get_wtime() - total_elt);

  if (debug) {
    printf("%d -- generate_bter_MPI() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}
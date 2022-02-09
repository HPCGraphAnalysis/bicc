/*
//@HEADER
// *****************************************************************************
//
//  HPCGraph: Graph Computation on High Performance Computing Systems
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/

#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

void throw_err(char const* err_message)
{
  fprintf(stderr, "Error: %s\n", err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task)
{
  fprintf(stderr, "Task %d Error: %s\n", task, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task, int32_t thread)
{
  fprintf(stderr, "Task %d Thread %d Error: %s\n", task, thread, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void quicksort_dec(uint64_t* arr1, uint64_t* arr2, int64_t left, int64_t right) 
{
  int64_t i = left;
  int64_t j = right;
  uint64_t temp; uint64_t temp2;
  uint64_t pivot = arr1[(left + right) / 2];

  while (i <= j) 
  {
    while (arr1[i] > pivot) {i++;}
    while (arr1[j] < pivot) {j--;}
  
    if (i <= j) 
    {
      temp = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = temp;
      temp2 = arr2[i];
      arr2[i] = arr2[j];
      arr2[j] = temp2;
      ++i;
      --j;
    }
  }

  if (j > left)
    quicksort_dec(arr1, arr2, left, j);
  if (i < right)
    quicksort_dec(arr1, arr2, i, right);
}

void quicksort(uint64_t* arr1, int64_t left, int64_t right) 
{
  int64_t i = left;
  int64_t j = right;
  uint64_t temp;
  uint64_t pivot_index = (left + right) / 2;
  uint64_t pivot = arr1[pivot_index];

  while (i <= j) 
  {
    while (arr1[i] < pivot) {i += 1;}
    while (arr1[j] > pivot) {j -= 1;}
  
    if (i <= j) 
    {
      temp = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = temp;
      i += 1;
      j -= 1;
    }
  }

  if (j > left)
    quicksort(arr1, left, j);
  if (i < right)
    quicksort(arr1, i, right);
}

void quicksort_inc(uint64_t* arr1, int64_t left, int64_t right) 
{
  int64_t i = left;
  int64_t j = right;
  uint64_t temp;
  uint64_t pivot_index = (left/3 + right/3) / 2 * 3;
  uint64_t pivot = arr1[pivot_index];

  while (i <= j) 
  {
    while (arr1[i] < pivot) {i += 3;}
    while (arr1[j] > pivot) {j -= 3;}
  
    if (i <= j) 
    {
      temp = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = temp;
      temp = arr1[i+1];
      arr1[i+1] = arr1[j+1];
      arr1[j+1] = temp;
      temp = arr1[i+2];
      arr1[i+2] = arr1[j+2];
      arr1[j+2] = temp;
      i += 3;
      j -= 3;
    }
  }

  if (j > left)
    quicksort_inc(arr1, left, j);
  if (i < right)
    quicksort_inc(arr1, i, right);
}

uint64_t* str_to_array(char *input_list_str, uint64_t* num)
{
  char *cp = strtok(input_list_str, ",");
  if (cp == NULL) {
    return (uint64_t*)malloc((*num)*sizeof(uint64_t));
  }

  int64_t my_index = -1;
  uint64_t n;
  if (sscanf(cp, "%lu", &n) == 1) {
      my_index = (int64_t)*num;
      *num += 1;
  } else {
      printf("Invalid integer token '%s'\n", cp);
  }
  uint64_t *array = str_to_array(NULL, num);
  if (my_index >= 0) {
      array[my_index] = n;
  }
  return array;
}

void parallel_prefixsums(
  uint64_t* in_array, uint64_t* out_array, uint64_t size)
{
  uint64_t* thread_sums;

#pragma omp parallel
{
  uint64_t nthreads = (uint64_t)omp_get_num_threads();
  uint64_t tid = (uint64_t)omp_get_thread_num();
#pragma omp single
{
  thread_sums = new uint64_t[nthreads+1];
  thread_sums[0] = 0;
}

  uint64_t my_sum = 0;
#pragma omp for schedule(static)
  for(uint64_t i = 0; i < size; ++i) {
    my_sum += in_array[i];
    out_array[i] = my_sum;
  }

  thread_sums[tid+1] = my_sum;
#pragma omp barrier

  uint64_t my_offset = 0;
  for(uint64_t i = 0; i < (tid+1); ++i)
    my_offset += thread_sums[i];

#pragma omp for schedule(static)
  for(uint64_t i = 0; i < size; ++i)
    out_array[i] += my_offset;
}

  delete [] thread_sums;
}
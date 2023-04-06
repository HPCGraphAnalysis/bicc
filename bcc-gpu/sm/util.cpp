
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

void parallel_prefixsums(long* in_array, long* out_array, long size)
{
  long* thread_sums;

#pragma omp parallel
{
  long nthreads = (long)omp_get_num_threads();
  long tid = (long)omp_get_thread_num();
#pragma omp single
{
  thread_sums = new long[nthreads+1];
  thread_sums[0] = 0;
}

  long my_sum = 0;
#pragma omp for schedule(static)
  for(int i = 0; i < size; ++i) {
    my_sum += in_array[i];
    out_array[i] = my_sum;
  }

  thread_sums[tid+1] = my_sum;
#pragma omp barrier

  long my_offset = 0;
  for(int i = 0; i < (tid+1); ++i)
    my_offset += thread_sums[i];

#pragma omp for schedule(static)
  for(int i = 0; i < size; ++i)
    out_array[i] += my_offset;
}

  delete [] thread_sums;
}

void quicksort(int* arr, int left, int right) 
{
  int i = left;
  int j = right;
  int temp = -1;
  int pivot = arr[(left + right) / 2];

  while (i <= j) 
  {
    while (arr[i] < pivot) {i++;}
    while (arr[j] > pivot) {j--;}
  
    if (i <= j) 
    {
      temp = arr[i];
      arr[i] = arr[j];
      arr[j] = temp;
      ++i;
      --j;
    }
  }

  if (j > left)
    quicksort(arr, left, j);
  if (i < right)
    quicksort(arr, i, right);
}
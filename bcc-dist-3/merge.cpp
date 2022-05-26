
for (uint64_t curr_size = 1; curr_size <= num_this_rank - 1; curr_size *= 2) {
#pragma omp for
  for (left_start=0; left_start<n-1; left_start += 2*curr_size)
  {
    uint64_t mid = min(left_start + curr_size - 1, num_this_rank - 1);
    uint64_t right_end = min(left_start + 2*curr_size - 1, num_this_rank - 1);
       
    uint64_t i, j, k;
    uint64_t n1 = mid - left_start + 1;
    uint64_t n2 = right_end - mid;
    uint64_t* L = new uint64_t[n1];
    uint64_t* R = new uint64_t[n2];
    memcpy(L, &arr[left_start], n1*sizeof(uint64_t));
    memcpy(R, &arr[mid + 1], n2*sizeof(uint64_t));
 
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t k = left_start;
    while (i < n1 && j < n2) {
      if (L[i] <= R[j]) {
        arr[k] = L[i];
        ++i;
      } else {
        arr[k] = R[j];
        ++j;
      }
      ++k;
    }
 
    while (i < n1) {
      arr[k] = L[i];
      i++;
      k++;
    }
 
    while (j < n2) {
      arr[k] = R[j];
      j++;
      k++;
    }
  }
}

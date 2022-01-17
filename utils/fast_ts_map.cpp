
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "fast_ts_map.h"

inline uint64_t hash64(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

void init_map(fast_ts_map* map, uint64_t init_size)
{
  map->arr = (entry*)malloc(init_size*sizeof(entry));
  if (map->arr == NULL) {
    printf("init_map(), unable to allocate resources\n");
    exit(0);
  }

  map->capacity = init_size;
 
#pragma omp parallel for
  for (uint64_t i = 0; i < map->capacity; ++i) {
    map->arr[i].key = NULL_KEY64;
    map->arr[i].val = NULL_KEY64;
    map->arr[i].test = false;
  }
}

void clear_map(fast_ts_map* map)
{
  free(map->arr);
  map->capacity = 0;
}

int64_t test_set_value(fast_ts_map* map, 
  uint32_t src, uint32_t dst, uint64_t val)
{
  uint64_t key = (((uint64_t)src << 32) | (uint64_t)dst);
  uint64_t init_idx = hash64(key) % map->capacity;
  
  if (src == dst) {
    printf("ERROR %u %u %lu\n", src, dst, key);
    exit(0);
  }

  for (uint64_t idx = init_idx;; idx = (idx+1) % map->capacity) {
    bool test = false;
    //test = __sync_fetch_and_or(&map->arr[idx].val, true);
#pragma omp atomic capture
    { test = map->arr[idx].test; map->arr[idx].test = true; }

    // race condition handling below
    // - other thread which won slot might have not yet set key
    if (test == false) { // this thread got empty slot
      map->arr[idx].key = key;
      map->arr[idx].val = val;
      return -1;
    }
    else if (test == true) {// key already exists in table
      // wait for key to get set if it isn't yet
      while (map->arr[idx].key == NULL_KEY64) {
        printf("%u %u %lu %lu\n", src, dst, val, map->arr[idx].key); // can comment this out and do other trivial work
      }

      if (map->arr[idx].key == key) // this key already exists in table
        return (int64_t)idx;
    } // else slot is taken by another key, loop and increment
  }

  return -1;
}

uint64_t get_value(fast_ts_map* map, uint32_t src, uint32_t dst)
{
  uint64_t key = (((uint64_t)src << 32) | (uint64_t)dst);
  uint64_t init_idx = hash64(key) % map->capacity;

  
  for (uint64_t idx = init_idx;; idx = (idx+1) % map->capacity) {
    if (map->arr[idx].key == key)
      return map->arr[idx].val;
    else if (map->arr[idx].key == NULL_KEY64)
      return NULL_KEY64;
    else if (idx == init_idx - 1)
      return NULL_KEY64;
  }
  
  return 0;
}
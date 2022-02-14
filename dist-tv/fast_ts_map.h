#ifndef _FAST_TS_MAP_H_
#define _FAST_TS_MAP_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define NULL_KEY 18446744073709551615U

struct entry {
    uint64_t key;
    uint64_t val;
} ;

struct fast_ts_map {
    entry* arr;
    uint64_t capacity;
} ;

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
    map->arr[i].key = NULL_KEY;
    map->arr[i].val = NULL_KEY;
  }
}

void clear_map(fast_ts_map* map)
{
  free(map->arr);
  map->capacity = 0;
}

inline uint64_t hash64(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}

inline uint64_t test_set_value(fast_ts_map* map, uint32_t src, uint32_t dst, uint64_t val)
{
    uint64_t key = (((uint64_t)src << 32) | (uint64_t)dst);
    uint64_t init_idx = hash64(key) % map->capacity;

    for (uint64_t idx = init_idx;; idx = (idx+1) % map->capacity) {
      //bool test = false;
      uint64_t test;
      //test = __sync_fetch_and_or(&map->arr[idx].val, true);
      #pragma omp atomic capture
    { test = map->arr[idx].val; map->arr[idx].val = val; }

    // race condition handling below
    // - other thread which won slot might have not yet set key
    if (test == NULL_KEY) { // this thread got empty slot
      map->arr[idx].key = key;
      map->arr[idx].val = val;
      return val;
    }
    else if (test != NULL_KEY) {// key already exists in table
        // wait for key to get set if it isn't yet
      while (map->arr[idx].key == NULL_KEY) {
          //printf("."); // can comment this out and do other trivial work
      }

      if (map->arr[idx].key == key) // this key already exists in table
        return map->arr[idx].val;
    } // else slot is taken by another key, loop and increment
  }

  return NULL_KEY;
}

inline uint64_t get_value(fast_ts_map* map, uint32_t src, uint32_t dst){
  uint64_t key = (((uint64_t)src << 32) | (uint64_t)dst);
  uint64_t init_idx = hash64(key) % map->capacity;
  for(uint64_t idx = init_idx;; idx = (idx+1) % map->capacity){
    if(map->arr[idx].key == key){
      return map->arr[idx].val;
    } else if(map->arr[idx].key == NULL_KEY){
      return NULL_KEY;
    }
  }
}


#endif

#ifndef _FAST_TS_MAP_H_
#define _FAST_TS_MAP_H_

#define NULL_KEY 0
#define NULL_KEY32 4294967295U
#define NULL_KEY64 18446744073709551615UL

#include "stdint.h"

struct entry {
  uint64_t key;
  uint64_t val;
  bool test;
} ;

struct fast_ts_map {
  entry* arr;
  uint64_t capacity;
} ;

void init_map(fast_ts_map* map, uint64_t init_size);

void clear_map(fast_ts_map* map);

int64_t test_set_value(fast_ts_map* map, 
  uint32_t src, uint32_t dst, uint64_t val);

uint64_t get_value(fast_ts_map* map, uint32_t src, uint32_t dst);



#endif

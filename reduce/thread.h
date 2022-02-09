#ifndef __thread_h__
#define __thread_h__

#define THREAD_H_QUEUE_SIZE 1024

inline void add_to_queue(uint64_t* thread_queue, uint64_t& thread_queue_size, 
                         uint64_t* queue_next, uint64_t& queue_size_next, 
                         uint64_t vert1, uint64_t vert2);
inline void empty_queue(uint64_t* thread_queue, uint64_t& thread_queue_size, 
                        uint64_t* queue_next, uint64_t& queue_size_next);


inline void add_to_queue(uint64_t* thread_queue, uint64_t& thread_queue_size, 
                         uint64_t* queue_next, uint64_t& queue_size_next,  
                         uint64_t vert1, uint64_t vert2)
{
  thread_queue[thread_queue_size++] = vert1;
  thread_queue[thread_queue_size++] = vert2;

  if (thread_queue_size == THREAD_H_QUEUE_SIZE)
    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
}

inline void empty_queue(uint64_t* thread_queue, uint64_t& thread_queue_size, 
                        uint64_t* queue_next, uint64_t& queue_size_next)
{
  uint64_t start_offset;

#pragma omp atomic capture
  start_offset = queue_size_next += thread_queue_size;

  start_offset -= thread_queue_size;
  for (uint64_t i = 0; i < thread_queue_size; ++i)
    queue_next[start_offset + i] = thread_queue[i];
  thread_queue_size = 0;
}


#endif
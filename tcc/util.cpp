
#include <stdlib.h>
#include <sys/time.h>

double timer() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}
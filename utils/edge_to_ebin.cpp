#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <unordered_map>

void read_edges(char* filename,
  long& num_edges, long*& srcs, long*& dsts)
{
  FILE* infile = fopen(filename, "r");
  char line[256];

  long count = 0;
  long cur_size = 1024*1024;
  srcs = (long*)malloc(cur_size*sizeof(long));
  dsts = (long*)malloc(cur_size*sizeof(long));

  while(fgets(line, 256, infile) != NULL) {
    if (line[0] == '%') continue;
    if (line[0] == '#') continue;

    sscanf(line, "%li %li", &srcs[count], &dsts[count]);
    ++count;
    
    if (count > cur_size) {
      cur_size *= 2;
      srcs = (long*)realloc(srcs, cur_size*sizeof(long));
      dsts = (long*)realloc(dsts, cur_size*sizeof(long));
    }
  }  
  num_edges = count;

  printf("Edges: %li\n", num_edges);

  fclose(infile);

  return;
}


void write_ebin(char* filename, long num_edges, long* srcs, long* dsts)
{
  FILE* outfile = fopen(filename, "wb");
  
  uint32_t write[2];
  for (int i = 0; i < num_edges; ++i) {
    write[0] = (uint32_t)srcs[i];
    write[1] = (uint32_t)dsts[i];
    fwrite(write, sizeof(uint32_t), 2, outfile);
  }
  
  fclose(outfile);
}

int main(int argc, char* argv[])
{
  setbuf(stdout, NULL);
  
  long num_verts = 0;
  long num_edges = 0;
  long* srcs = NULL;
  long* dsts = NULL;
  
  read_edges(argv[1], num_edges, srcs, dsts);
  write_ebin(argv[2], num_edges, srcs, dsts);
  
  free(srcs);
  free(dsts);
  
  return 0;
}


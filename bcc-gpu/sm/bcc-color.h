#ifndef _BCC_H_
#define _BCC_H_

#include "graph.h"

#define BLOCK_SIZE 512

int bcc_color_decomposition(graph* g_host, int* bcc_maps, int& bcc_count, int& art_vert_count);

#endif

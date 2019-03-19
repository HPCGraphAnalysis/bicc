#include<iostream>
#include<string>
#include<cstdlib>
using std::string;
using std::cout;

void print_usage(){
  cout<<"Generator arguments:\n \
         \targv[1]: integer >= 2 representing both dimensions of the central ice block\n \
         \t         (central ice block will be a regular grid of argv[0]xargv[0] nodes)\n \
         \targv[2]: integer >= 2 representing both dimensions of the degenerate ice blocks\n \
         \targv[3]: integer >= 0 representing number of degenerate ice chains\n \
         \targv[4]: integer >= 0 representing how many degenerate features are chained together\n \
         \t         (the length of chains of degenerate features off of the central ice block)\n \
         \targv[5]: integer >= 0 representing number of complex features\n \
         \t         (complex features are chains of ice blocks that connect back to the central ice block)\n \
         \targv[6]: integer >= 1 representing how many degenerate blocks are chained together in complex chains\n \
         \targv[7]: integer >= 0 representing how many nodes on the central ice sheet are grounded to begin with\n \
         \t         (capped at argv[0]*argv[0])\n \
         \targv[8]: filename for the output mesh\n";
}

int main(int argc, char** argv) {
  
  /* 
    Generator arguments:
      argv[0]: integer >= 2 representing both dimensions of the central ice block
               (central ice block will be a regular grid of argv[0] x argv[0] nodes)
      argv[1]: integer >= 2 representing both dimensions of the degenerate ice blocks
      argv[2]: integer >= 0 representing number of degenerate features
      argv[3]: integer <= argv[2] representing number of complex features 
               (complex features are chains of ice blocks that connect back to the central ice block)
      argv[4]: integer <= argv[2] representing how many degenerate features are chained together
               (some measure of complexity)
      argv[5]: integer >= 0 representing how many nodes on the central ice sheet are grounded to begin with
               (capped at arg[0]*arg[0])
      argv[6]: filename for the output mesh 
     
    Notes for total node count:
      - All nodes in the central ice block get created.
      - All nodes except for one in degenerate blocks must be created,
        as degenerate blocks hang off of pre-established ice
      - All nodes except for two in the final block of a complex feature must be created,
        since those blocks connect to two pre-existing ice blocks.
      - formula:
          (central block) +        (degenerate features)          +      (complex features)
          argv[1]*argv[1] + (argv[2]*argv[2]-1)*(argv[3]) + (argv[2]*argv[2]-2)*(argv[4])
  */
  int central_ice_size = 1000;
  int degenerate_block_size = 10;
  int num_degenerate_chains = 15;
  int degenerate_chain_length = 3;
  int num_complex_chains = 10;
  int complex_chain_length = 3;
  int initially_grounded = 500;
  string filename = "mesh.quad.msh";
  if(argc < 9) 
    print_usage();
  else { 
    //for(int i = 1; i < argc; i++) {
    //  cout<<"argv["<<i<<"] = "<<argv[i]<<"\n";
    //}
    central_ice_size = atoi(argv[1]);
    degenerate_block_size = atoi(argv[2]);
    num_degenerate_chains = atoi(argv[3]);
    degenerate_chain_length = atoi(argv[4]);
    num_complex_chains = atoi(argv[5]);
    complex_chain_length = atoi(argv[6]);
    initially_grounded = atoi(argv[7]);
    filename = argv[8]; 
    
  }
  int total_vtx_count = central_ice_size*central_ice_size+(degenerate_block_size*degenerate_block_size - 1)*degenerate_chain_length*num_degenerate_chains + ((degenerate_block_size*degenerate_block_size-1)*(complex_chain_length-1) + (degenerate_block_size*degenerate_block_size-2))*num_complex_chains;

  cout<<"number of generated vertices: "<<total_vtx_count<<"\n";
  return 0;
}

#include<iostream>
#include<string>
#include<fstream>
#include<cstdlib>
#include<random>

using std::string;
using std::cout;
using std::ofstream;

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


void writeFileHeaders(ofstream& mesh, ofstream& borders, ofstream& basal,int nverts, int nelements, int nborders){
  //write two random strings to the mesh files/
  mesh<<"generated mesh\n";
  borders<<"generated borders\n";
  //write the headers to the mesh files, and the first line to the basal friction file
  mesh<<nverts<<" "<<nelements<<" 0\n";
  for(int i = 1; i <= nverts; i++){
    mesh<<i<<"\n";
  }
  borders<<"0 0 "<<nborders<<"\n\n";
  basal<<nverts<<"\n";
}

void generateGrounding(bool& grounded_two, std::default_random_engine& gen, std::uniform_real_distribution<double>& dist,
		       float threshold, int groundable_verts,int total_verts, ofstream& groundfile){
  int num_grounded = 0;
  for(int i = 0; i < groundable_verts; i++){
    double roll = dist(gen);
    num_grounded += (roll<= threshold);
    groundfile<<(roll<= threshold)<<"\n";
  }
  grounded_two = (num_grounded >= 2);
  for(int i=groundable_verts; i< total_verts; i++){
    groundfile<<"0\n";
  }
}

void generateDegenerateFeatures(int groundable_verts, int degenerate_block_size, int chain_length, int num_chains,
		                ofstream& meshfile, ofstream& borderfile, std::default_random_engine& gen){

  std::uniform_int_distribution<int> vtxDist(0, groundable_verts-1);

  int last_block_start = 0;
  int last_block_end = groundable_verts;
  //for all chains
  for(int c = 0; c < num_chains; c++){
    int attachvtx = vtxDist(gen);
    for(int chain = 0; chain < chain_length; chain++){
      
      //top border edges for this block
      borderfile<<attachvtx+1<<" "<<last_block_end+1<<" 0\n";
      for(int i = 0; i <degenerate_block_size-2; i++){
        borderfile<<last_block_end+i+1<<" "<<last_block_end+i+1+1<<" 0\n";
      }

      //left borders for this block
      borderfile<<attachvtx+1<<" "<<last_block_end+degenerate_block_size-1+1<<" 0\n";
      for(int i = degenerate_block_size-1; i < (degenerate_block_size*(degenerate_block_size - 2)); i += degenerate_block_size){
        borderfile<<last_block_end+i+1<<" "<<last_block_end + i + degenerate_block_size+1<<" 0\n";
      }

      //right borders for this block
      for(int i = degenerate_block_size-2; i < degenerate_block_size*(degenerate_block_size-1); i += degenerate_block_size){
        borderfile<<last_block_end+i+1<<" "<<last_block_end+i+degenerate_block_size+1<<" 0\n";
      }

      //bottom borders for this block
      for(int i=(degenerate_block_size-1)*(degenerate_block_size)-1; i < degenerate_block_size*degenerate_block_size-2; i++){
        borderfile<<last_block_end+i+1<<" "<<last_block_end+i+1+1<<" 0\n";
      }

      //first element attaches to the previous block
      meshfile<<attachvtx+1<<" "<<last_block_end+1<<" "<<last_block_end + degenerate_block_size+1<<" ";
      meshfile<<last_block_end+degenerate_block_size - 1+1<<" 0\n";
      //create the other elements in this row
      for(int j =0; j<degenerate_block_size-2; j++){
        meshfile<<last_block_end+ j+1<<" "<<last_block_end+j+1+1<<" "<<last_block_end+degenerate_block_size+j+1+1<<" ";
	meshfile<<last_block_end + degenerate_block_size + j+1<<" 0\n";
      }
      //create the other elements in this block
      for(int i=1; i < degenerate_block_size-1; i++){
        for(int j=0; j < degenerate_block_size-1; j++){
	  meshfile<<last_block_end+i*degenerate_block_size+j-1+1<<" "<<last_block_end+i*degenerate_block_size+j+1<<" ";
	  meshfile<<last_block_end+(i+1)*degenerate_block_size+j+1<<" "<<last_block_end+(i+1)*degenerate_block_size+j-1+1<<" 0\n";
	}
      }
      
      last_block_start = last_block_end;
      last_block_end += degenerate_block_size*degenerate_block_size - 1;
      attachvtx = vtxDist(gen)%(degenerate_block_size*degenerate_block_size - 1) + last_block_start;
    }
  }
}


void generateComplexChains(int central_ice_size, int complex_block_size, int chain_length, int num_complex_chains,
		          ofstream& meshfile, ofstream& borderfile, std::default_random_engine& gen){
  std::uniform_int_distribution<int> vtxDist(0,central_ice_size-1);
  //for all the chains
  for(int c = 0; c < num_complex_chains; c++){
    //get the starting point randomly
    int first_attachment_vertex = vtxDist(gen);
    //get the ending point randomly, assure it is not the starting point.
    int second_attachment_vertex = vtxDist(gen);
    while(first_attachment_vertex == second_attachment_vertex) second_attachment_vertex = vtxDist(gen);
    
    //these are the ranges of vertices to generate connection vertices.
    int last_block_start = central_ice_size;
    int last_block_end = central_ice_size + complex_block_size*complex_block_size - 1;

    //create the first element using the starting vertex
    meshfile<< first_attachment_vertex+1<<" "<<central_ice_size+1<<" "<<central_ice_size+complex_block_size+1<<" ";
    meshfile<<central_ice_size+complex_block_size-1+1<<" 0\n";

    //cout<<"Borders:\n";
    //add the associated borders to the border file
    //top borders for this block
    borderfile<<first_attachment_vertex+1<<" "<<central_ice_size+1<<" 0\n";
    for(int i = 0; i <complex_block_size-2; i++){
      borderfile<<central_ice_size+i+1<<" "<<central_ice_size+i+1+1<<" 0\n";
    }

    //left borders for this block
    borderfile<<first_attachment_vertex+1<<" "<<central_ice_size+complex_block_size-1+1<<" 0\n";
    for(int i = complex_block_size-1; i < (complex_block_size*(complex_block_size - 2)); i += complex_block_size){
      borderfile<<central_ice_size+i+1<<" "<<central_ice_size+i+complex_block_size+1<<" 0\n";
    }

    //right borders for this block
    for(int i = complex_block_size-2; i <  complex_block_size*(complex_block_size-1); i+=complex_block_size){
      borderfile<<central_ice_size+i+1<<" "<<central_ice_size+i+complex_block_size+1<<" 0\n";
    }

    //bottom borders for this block
    for(int i = (complex_block_size-1)*(complex_block_size)-1; i< complex_block_size*complex_block_size-2; i++){
      borderfile<<central_ice_size+i+1<<" "<<central_ice_size+i+1+1<<" 0\n";
    } 

    //create the rest of the elements in this row
    for(int j=0; j < complex_block_size-2; j++){
      meshfile<<central_ice_size + j+1<<" "<<central_ice_size+j+1+1<<" "<<central_ice_size+complex_block_size+j+1+1<<" ";
      meshfile<<central_ice_size + complex_block_size + j+1<<" 0\n";
    }
    //create the rest of the elements in this complex block
    for(int i=1; i < complex_block_size-1; i++){
      for(int j=0; j < complex_block_size-1; j++){
        meshfile<<central_ice_size + i*complex_block_size+j-1+1<<" "<<central_ice_size + i*complex_block_size+j+1<<" ";
	meshfile<<central_ice_size + (i+1)*complex_block_size+j+1<<" "<<central_ice_size + (i+1)*complex_block_size+j-1+1<<" 0\n";
      }
    }
    //cout<<"\n\nMiddle blocks:\n";
    //create the middle blocks using a starting vertex from the previous block
    for(int chain = 0; chain < chain_length; chain++){
      int attachvtx = vtxDist(gen)%(complex_block_size*complex_block_size-1) + last_block_start;

      //top border edges for this block
      borderfile<<attachvtx+1<<" "<<last_block_end+1<<" 0\n";
      for(int i = 0; i <complex_block_size-2; i++){
        borderfile<<last_block_end+i+1<<" "<<last_block_end+i+1+1<<" 0\n";
      }

      //left borders for this block
      borderfile<<attachvtx+1<<" "<<last_block_end+complex_block_size-1+1<<" 0\n";
      for(int i = complex_block_size-1; i < (complex_block_size*(complex_block_size - 2)); i += complex_block_size){
        borderfile<<last_block_end+i+1<<" "<<last_block_end + i + complex_block_size+1<<" 0\n";
      }

      //right borders for this block
      for(int i = complex_block_size-2; i < complex_block_size*(complex_block_size-1); i += complex_block_size){
        borderfile<<last_block_end+i+1<<" "<<last_block_end+i+complex_block_size+1<<" 0\n";
      }

      //bottom borders for this block
      for(int i=(complex_block_size-1)*(complex_block_size)-1; i < complex_block_size*complex_block_size-2; i++){
        borderfile<<last_block_end+i+1<<" "<<last_block_end+i+1+1<<" 0\n";
      }
      
      //first element attaches to the previous block
      meshfile<<attachvtx+1<<" "<<last_block_end+1<<" "<<last_block_end + complex_block_size+1<<" ";
      meshfile<<last_block_end+complex_block_size - 1+1<<" 0\n";
      //create the other elements in this row
      for(int j =0; j<complex_block_size-2; j++){
        meshfile<<last_block_end+ j+1<<" "<<last_block_end+j+1+1<<" "<<last_block_end+complex_block_size+j+1+1<<" ";
	meshfile<<last_block_end + complex_block_size + j+1<<" 0\n";
      }
      //create the other elements in this block
      for(int i=1; i < complex_block_size-1; i++){
        for(int j=0; j < complex_block_size-1; j++){
	  meshfile<<last_block_end + i*complex_block_size + j - 1+1<<" "<<last_block_end + i*complex_block_size + j+1<<" ";
	  meshfile<<last_block_end + (i+1)*complex_block_size + j+1<<" "<<last_block_end + (i+1)*complex_block_size + j - 1+1<<" 0\n";
	}
      }
      //update the generation ranges for the next complex block
      last_block_start+=complex_block_size*complex_block_size - 1;
      last_block_end += complex_block_size*complex_block_size - 1;
      //cout<<"\n";
    }
    //cout<<"\n\nLast complex block:\n";
    //create the last block using a starting vertex from the previous block, and an ending vertex from the center range.
    int attachvtx = vtxDist(gen)%(complex_block_size*complex_block_size-1) + last_block_start;

    //top border edges for this block
    borderfile<<attachvtx+1<<" "<<last_block_end+1<<" 0\n";
    for(int i = 0; i <complex_block_size-2; i++){
      borderfile<<last_block_end+i+1<<" "<<last_block_end+i+1+1<<" 0\n";
    }

    //left borders for this block
    borderfile<<attachvtx+1<<" "<<last_block_end+complex_block_size-1+1<<" 0\n";
    for(int i = complex_block_size-1; i < (complex_block_size*(complex_block_size - 2)); i += complex_block_size){
      borderfile<<last_block_end+i+1<<" "<<last_block_end + i + complex_block_size+1<<" 0\n";
    }

    //right borders for this block
    for(int i = complex_block_size-2; i < complex_block_size*(complex_block_size-2); i += complex_block_size){
      borderfile<<last_block_end+i+1<<" "<<last_block_end+i+complex_block_size+1<<" 0\n";
    }
    borderfile<<last_block_end+complex_block_size-2+complex_block_size*(complex_block_size-2)+1<<" "<<second_attachment_vertex+1<<" 0\n";

    //bottom borders for this block
    for(int i=(complex_block_size-1)*(complex_block_size)-1; i < complex_block_size*complex_block_size-3; i++){
      borderfile<<last_block_end+i+1<<" "<<last_block_end+i+1+1<<" 0\n";
    }
    borderfile<<last_block_end+complex_block_size*complex_block_size-3+1<<" "<<second_attachment_vertex+1<<" 0\n";

    //first element attaches to the previous block
    meshfile<<attachvtx+1<<" "<<last_block_end+1<<" "<<last_block_end + complex_block_size+1<<" ";
    meshfile<<last_block_end + complex_block_size - 1+1<<" 0\n";

    for(int j = 0; j < complex_block_size-2; j++){
      meshfile<<last_block_end + j+1<<" "<<last_block_end + j + 1+1<<" "<<last_block_end +complex_block_size + j +1+1<<" ";
      meshfile<<last_block_end + complex_block_size + j+1<<" 0\n";
    }

    for(int i = 1; i < complex_block_size-2; i++){
      for(int j = 0; j < complex_block_size-1; j++){
        meshfile<<last_block_end + i*complex_block_size + j - 1+1<<" "<<last_block_end + i*complex_block_size + j+1<<" ";
	meshfile<<last_block_end + (i+1)*complex_block_size + j+1<<" "<<last_block_end + (i+1)*complex_block_size + j - 1+1<<" 0\n";
      }
    }

    //the last element attaches to the second_attachment_vertex
    for(int j = 0; j < complex_block_size-2; j++){
      meshfile<<last_block_end + (complex_block_size-2)*complex_block_size + j -1+1<<" ";
      meshfile<<last_block_end + (complex_block_size-2)*complex_block_size + j+1 <<" ";
      meshfile<<last_block_end + (complex_block_size-1)*complex_block_size + j+1 <<" ";
      meshfile<<last_block_end + (complex_block_size-1)*complex_block_size + j - 1+1<<" 0\n";
    }
    
    
    //the last element connects the last three vertices to the second attachment vertex
    meshfile<<last_block_end + (complex_block_size-2)*complex_block_size - 1 + complex_block_size-2+1<<" ";
    meshfile<<last_block_end + (complex_block_size-2)*complex_block_size - 1 + complex_block_size-1+1<<" ";
    meshfile<<second_attachment_vertex+1<<" ";
    meshfile<<last_block_end + (complex_block_size-1)*complex_block_size - 1 + complex_block_size-2+1<<" 0\n";
    
    //update the initial offset so the next chain's vertex IDs are correct.
    central_ice_size = last_block_end + complex_block_size*complex_block_size - 2;
  }
}

void generateCenter(int dim, ofstream& mesh, ofstream& borders){
  
  for(int i = 0; i < dim-1; i++){
    for(int j = 0; j < dim-1; j++){
      mesh<<i*dim+j+1<<" "<<i*dim+j+1+1<<" "<<(i+1)*dim+j+1+1<<" "<<(i+1)*dim+j+1<<" 0\n";
    }
  }

  //cout<<"Boundary Edge nodes:\n";
  //cout<<"\t top row: ";
  for(int i = 0; i < dim-1; i++){
    borders<<i+1<<" "<<i+1+1<<" 0\n";
  }
  //cout<<"\n";
  //cout<<"\t left side: ";
  for(int i = 0; i < dim*dim - dim; i+=dim){
    borders<<i+1<<" "<<i+dim+1<<" 0\n";
  }
  //cout<<"\n";
  //cout<<"\t bottom side: ";
  for(int i = dim*(dim-1); i < dim*dim-1; i++){
    borders<<i+1<<" "<<i+1+1<<" 0\n";
  }
  //cout<<"\n";
  //cout<<"\t right side: ";
  for(int i = dim-1;  i < dim*dim - dim; i+= dim){
    borders<<i+1<<" "<<i+dim+1<<" 0\n";
  }
  //cout<<"\n";
}

//this might be off by one in the degenerate feature case.
void generateAnswers(bool grounded_two, int groundable_verts, int total_verts, ofstream& answers){
  if(grounded_two){
    //only remove degenerate features
    for(int i = groundable_verts; i < total_verts; i++){
      answers<<i+1<<"\n";
    }
  } else {
    //remove all vertices
    for(int i = 1; i <= total_verts; i++){
      answers<<i<<"\n";
    }
  }
}

int main(int argc, char** argv) {
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  int central_ice_size = 1000;
  int degenerate_block_size = 10;
  int complex_block_size = 10;
  int num_degenerate_chains = 15;
  int degenerate_chain_length = 3;
  int num_complex_chains = 10;
  int complex_chain_length = 3;
  float initially_grounded = 0.5;
  string filename = "mesh";
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
    complex_block_size = atoi(argv[5]);
    num_complex_chains = atoi(argv[6]);
    complex_chain_length = atoi(argv[7]);
    initially_grounded = atof(argv[8]);
    filename = argv[9]; 
    
  }
 
  //need to write out the mesh file, <filename>.quad.msh
  //the border file, <filename>-borders.quad.msh
  //the basal friction file, <filename>-basal-friction.quad.msh
  //and the answer file(s), <filename>-s1-answers.txt
  ofstream meshfile(filename+".quad.msh");
  ofstream borderfile(filename+"-borders.quad.msh");
  ofstream groundfile(filename+"-basal-friction.ascii");
  ofstream answerfile(filename+"-answer.txt");

  //important values
  int degenerate_block  = (degenerate_block_size*degenerate_block_size) - 1;
  int degenerate_chains = degenerate_block*degenerate_chain_length*num_degenerate_chains;
  int end_block = (complex_block_size*complex_block_size) - 2;
  int initial_block = (complex_block_size*complex_block_size) - 1;
  int complex_chain = initial_block * (complex_chain_length+1) + end_block;
  int complex_chains = complex_chain * num_complex_chains;
  int central_block = central_ice_size * central_ice_size;
  int total_vertices = central_block + degenerate_chains + complex_chains;
  int border_edges_central = (central_ice_size-1)*4; 
  int border_edges_complex = (complex_block_size-1)*4*(complex_chain_length+2)*num_complex_chains; 
  int border_edges_degenerate = (degenerate_block_size-1)*4*degenerate_chain_length*num_degenerate_chains;
  int border_edges_total = border_edges_central + border_edges_complex + border_edges_degenerate;
  int central_elements = (central_ice_size-1)*(central_ice_size-1);
  int complex_elements = ((complex_block_size-1)*(complex_block_size-1))*(complex_chain_length+2)*num_complex_chains; 
  int degenerate_elements = ((degenerate_block_size-1)*(degenerate_block_size-1))*degenerate_chain_length*num_degenerate_chains;
  int total_elements = central_elements + complex_elements + degenerate_elements;
  
  cout<<"writing file headers\n";
  writeFileHeaders(meshfile,borderfile,groundfile,total_vertices,total_elements,border_edges_total);

  cout<<"generating central ice\n";
  generateCenter(central_ice_size,meshfile,borderfile);
  
  generateComplexChains(central_block,complex_block_size,complex_chain_length,num_complex_chains, meshfile,borderfile,generator);
  
  
  
  int groundable_verts = central_block + complex_chains;
  
  generateDegenerateFeatures(groundable_verts, degenerate_block_size, degenerate_chain_length, num_degenerate_chains, meshfile,borderfile, generator);
  //if vtxID is < groundable_verts, it has an initially_grounded% chance to become grounded 
  cout<<"generating grounding information\n";
  bool grounded_two = false;
  generateGrounding(grounded_two, generator, distribution, initially_grounded, groundable_verts,total_vertices, groundfile);
  
  generateAnswers(grounded_two, groundable_verts, total_vertices, answerfile);

  meshfile.close();
  borderfile.close();
  groundfile.close();
  answerfile.close();

  cout<<"mesh file written out on "<<filename<<".quad.msh\n";
  cout<<"border edge file written on "<<filename<<"-borders.quad.msh\n";
  cout<<"basal friction file written on "<<filename<<"-basal-friction.ascii\n";
  cout<<"answer written on "<<filename<<"-answer.txt\n";
  return 0;
}

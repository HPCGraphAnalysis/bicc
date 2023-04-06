CXX = g++
CXXFLAGS = -fopenmp -Wextra -O0 -g -std=c++11
LINKFLAGS = -L/usr/local/cuda/lib64 -lcudart
CUDA = nvcc
CUDAFLAGS = -g -G -O0 -arch=sm_70
TARGET = bcc
TOCOMPILE = bcc-color.o graph.o util.o main.o


all: $(TOCOMPILE) 
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TOCOMPILE) $(LINKFLAGS) 

.cpp.o: 
	$(CXX) $(CXXFLAGS) -c $*.cpp

bcc-color.o: bcc-color.cu
	$(CUDA) $(CUDAFLAGS) -c bcc-color.cu

clean:
	rm -f *.o $(TARGET) 


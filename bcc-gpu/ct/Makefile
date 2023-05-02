CXX = g++
CXXFLAGS = -fopenmp -Wextra -O0 -g -std=c++11
LINKFLAGS = -L/usr/local/cuda/lib64 -lcudart
CUDA = nvcc
CUDAFLAGS = -g -G -O0 -arch=sm_70
TARGET = reduce
TOCOMPILE = graph.o util.o reduce.o main.o


all: $(TOCOMPILE) 
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TOCOMPILE) $(LINKFLAGS) 

.cpp.o: 
	$(CXX) $(CXXFLAGS) -c $*.cpp

reduce.o: reduce.cu
	$(CUDA) $(CUDAFLAGS) -c reduce.cu

clean:
	rm -f *.o $(TARGET) 


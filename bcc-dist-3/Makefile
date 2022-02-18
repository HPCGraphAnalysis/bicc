CXX = mpicxx
CXXFLAGS = -fopenmp -O0 -g -Wall -std=c++11
LINKFLAGS = -fopenmp -O0 -g -Wall -std=c++11
TARGET = bicc-dist
TOCOMPILE = util.o fast_map.o dist_graph.o comms.o io_pp.o wbter.o bicc.o reduce_graph.o preorder_tree.o pre_comms.o min_max_size.o main.o


all: $(TOCOMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f $(TOCOMPILE) $(TARGET)


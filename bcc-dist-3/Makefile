CXX = mpicxx
CXXFLAGS = -fopenmp -O3 -Wall -std=c++11
LINKFLAGS = -fopenmp -O3 -Wall -std=c++11
TARGET = bicc-dist
TOCOMPILE = util.o fast_map.o dist_graph.o comms.o io_pp.o wbter.o bicc.o main.o


all: $(TOCOMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f $(TOCOMPILE) $(TARGET)


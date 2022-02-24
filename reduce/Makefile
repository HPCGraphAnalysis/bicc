CXX = mpicxx
CXXFLAGS = -fopenmp -O0 -g -Wall -std=c++11
LINKFLAGS = -fopenmp -O0 -g -Wall -std=c++11
TARGET = reduce-graph
INCLUDE = ../include
INCLUDE_COMPILE = util.o fast_map.o dist_graph.o comms.o io_pp.o 
LOCAL_COMPILE = reduce_graph.o  main.o


all: $(INCLUDE_COMPILE) $(LOCAL_COMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(INCLUDE_COMPILE) $(LOCAL_COMPILE)

%.o: $(INCLUDE)/%.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDE)/$*.cpp

clean:
	rm -f $(INCLUDE_COMPILE) $(LOCAL_COMPILE) $(TARGET)


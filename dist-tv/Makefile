CXX = mpicxx
LINKFLAGS = -fopenmp -O0 -g -Wall -std=c++11
TARGET = dist-tv
INCLUDE = ../include
INCLUDE_COMPILE = util.o fast_map.o dist_graph.o comms.o io_pp.o
REDUCE = ../reduce
REDUCE_COMPILE = reduce_graph.o 
LOCAL_COMPILE =  generate.o bicc_dist.o main.o
CXXFLAGS = -fopenmp -O0 -g -Wall -std=c++11 -I$(INCLUDE) -I$(REDUCE)

all: $(INCLUDE_COMPILE) $(REDUCE_COMPILE) $(LOCAL_COMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(INCLUDE_COMPILE) $(REDUCE_COMPILE) $(LOCAL_COMPILE)

$(INCLUDE_COMPILE):
	cd $(INCLUDE) && $(CXX) $(CXXFLAGS) -c $*.cpp && cd ../dist-tv && mv $(INCLUDE)/$*.o ./

$(REDUCE_COMPILE): $(INCLUDE_COMPILE)
	cd $(REDUCE) && $(CXX) $(CXXFLAGS) -c $*.cpp && cd ../dist-tv && mv $(REDUCE)/$*.o ./

$(LOCAL_COMPILE): $(INCLUDE_COMPILE) $(REDUCE_COMPILE)
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean: 
	rm -f $(INCLUDE_COMPILE) $(REDUCE_COMPILE) $(LOCAL_COMPILE) $(TARGET)

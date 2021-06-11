MPICXX = mpicxx
CXXFLAGS = -fopenmp -g -Wall
LINKFLAGS = -fopenmp -g -Wall
TARGET = bicc
LIBTARGET = libbicc.a
TOCOMPILE = util.o generate.o fast_map.o dist_graph.o comms.o io_pp.o main.o
FORLIBBICC = util.o generate.o fast_map.o dist_graph.o comms.o io_pp.o bfs.o lca.o bicc_dist.o art_pt_heuristic.o


all: libbicc $(TOCOMPILE)
	$(MPICXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE) $(LIBTARGET)

libbicc: $(FORLIBBICC)
	ar rvs $(LIBTARGET) *.o

.cpp.o:
	$(MPICXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f *.o *.a $(TARGET)


CXX=g++
CXXFLAGS = -fopenmp -O0 -Wall -g
LINKFLAGS = -fopenmp -O0 -Wall -g
TARGET = tcc
TOCOMPILE = util.o tcc_bfs.o tcc_main.o

all: $(TOCOMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f *.o $(TARGET)

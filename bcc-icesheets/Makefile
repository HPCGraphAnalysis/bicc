CXX=g++
CXXFLAGS = -fopenmp -O3 -Wall
LINKFLAGS = -fopenmp -O3 -Wall
TARGET = bcc
TOCOMPILE = util.o bcc_bfs.o bcc_color.o bcc.o

all: $(TOCOMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f *.o $(TARGET)

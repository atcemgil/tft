# help from: http://edndoc.esri.com/arcobjects/9.0/arcgisdevhelp/developmentenvs/cpp/makefiles/makefile.linux.htm

PROGRAM = cudaconnect
INCLUDEDIRS = \
	-I/home/can/arastir/code/tensorver/cudaconnect/contrib/hiredis	

LIBDIRS = \
	-L/home/can/arastir/code/tensorver/cudaconnect/contrib/hiredis

LIBS = -lhiredis -ljsoncpp

LDFLAGS = $(LIBDIRS) $(LIBS)

CXXSOURCES = cudaconnect.cpp
CXXOBJECTS = $(CXXSOURCES:.cpp=.o)  # expands to list of object files
CXX = g++
CXXFLAGS = $(INCLUDEDIRS)

all: $(PROGRAM)

$(PROGRAM): $(CXXOBJECTS)
	$(CXX) -o $@ $(CXXOBJECTS) $(LDFLAGS)

clean:
	$(RM) -f $(CXXOBJECTS) $(PROGRAM)

cudaconnect.o: cudaconnect.cpp
	$(CXX) $(CXXFLAGS) -c -o cudaconnect.o cudaconnect.cpp

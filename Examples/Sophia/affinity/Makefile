#
# Definition of MACROS

CUDA = #/usr/local/cuda

CXX = mpicxx
CXXFLAGS = -g -fopenmp -O3 -std=c++0x #-I$(CUDA)/include

CUDA_CXX = nvcc
CUDA_CXXFLAGS = 

CPP = cpp -P -traditional
CPPFLAGS =

LD = $(CXX)
LIB = #-L$(CUDA)/lib64 -lcuda -lcudart

BINROOT=./
EX=hello_affinity
SHELL=/bin/sh

# -- subset of src files with cuda kernels
CUDA_SRC = 
CUDA_OBJ = $(CUDA_SRC:.cpp=.o)

SRC = $(filter-out $(CUDA_SRC), $(wildcard *.cpp))
INC = $(wildcard *.h)
OBJ = $(SRC:.cpp=.o)

#
# -- target : 	Dependencies
# --		Rule to create target

$(EX): 	$(OBJ) $(CUDA_OBJ)
	$(LD) -o $@ $(CXXFLAGS) $(OBJ) $(CUDA_OBJ) $(LIB)

####################################################################

$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: %.cpp
	$(CUDA_CXX) -x cu $(CUDA_CXXFLAGS) -c $< -o $@

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EX)

#
# -- Simple dependencies

$(OBJ) : $(INC)
$(CUDA_OBJ) : $(INC)

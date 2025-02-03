#
# Definition of MACROS

CXX = mpicxx
CXXFLAGS = 
#CXXFLAGS += -fiopenmp -fopenmp-targets=spir64
CXXFLAGS += -fiopenmp -fopenmp-targets=spir64_gen
#CXXFLAGS += -D_SINGLE_PRECISION

CPP = cpp -P -traditional
CPPFLAGS =

LD = $(CXX)
LDFLAGS =
LDFLAGS += -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device pvc"
LIB = 

BINROOT=./
EX=vecadd
SHELL=/bin/sh

SRC = $(wildcard *.cpp)
INC = $(wildcard *.h)
OBJ = $(SRC:.cpp=.o)

#
# -- target : 	Dependencies
# --		Rule to create target

$(EX): 	$(OBJ)
	$(LD) -o $@ $(LDFLAGS) $(OBJ) $(LIB)

####################################################################

$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<
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

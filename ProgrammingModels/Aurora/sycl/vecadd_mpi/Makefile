#
# Definition of MACROS

CXX = mpicxx
CXXFLAGS = --intel -fsycl #-fsycl-unnamed-lambda
CXXFLAGS += -fsycl-targets=spir64_gen

CPP = cpp -P -traditional
CPPFLAGS =

LD = $(CXX)
LDFLAGS = $(CXXFLAGS)
LDFLAGS += -Xsycl-target-backend "-device pvc"
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

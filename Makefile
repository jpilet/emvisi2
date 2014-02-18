CXXFLAGS=`pkg-config opencv --cflags` -O3 
LDLIBS=`pkg-config opencv --libs`

OBJ=fwncc.o ncc_proba.o growmat.o 
CC=g++

# if you want graphcut support, download Yuri Boykov's implementation
# http://www.adastral.ucl.ac.uk/~vladkolm/software/maxflow-v3.0.src.tar.gz
# and uncomment the two following lines
#OBJ+=graph.o  maxflow.o
#CXXFLAGS+=-DWITH_GRAPHCUT

PACKAGE=emvisi2

DIST_FILES= Makefile README emvisi2.cpp emvisi2.h \
	fwncc.cpp fwncc.h growmat.cpp growmat.h \
	imstat.h learn.cpp ncc_proba.cpp \
	data

all: emvisi2 learn

emvisi2: $(OBJ) emvisi2.cpp
	g++ -o $@ $^ -DTEST_EMVISI $(CXXFLAGS) $(LDLIBS)

learn: $(OBJ) learn.cpp emvisi2.cpp
	g++ -o $@ $^ $(CXXFLAGS) $(LDLIBS)

clean:
	rm -f $(OBJ) learn emvisi2 

dist:
	make -C data clean
	mkdir /tmp/$(PACKAGE)
	cp -r $(DIST_FILES) /tmp/$(PACKAGE)
	tar -czvf emvisi2.tar.gz -C /tmp $(PACKAGE)
	rm -fR /tmp/$(PACKAGE)

emvisi2.o: emvisi2.h fwncc.h imstat.h 
fwncc.o: fwncc.h
growmat.o: growmat.h
learn.o: fwncc.h emvisi2.h imstat.h growmat.h
emvisi2.o: fwncc.h imstat.h 

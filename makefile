USER_OBJS := AllAttributes.o  classifier.o detect.o dlibalign.o features.o io.o main.o prep.o sysutils.o utils.o

LIBS = $(shell pkg-config opencv --libs)

subdirs: 
	cd lbp && $(MAKE)
	cd ..


facialattributes: $(USER_OBJS)
	g++ -std=c++0x -o "MobileFacialAttributes" $(USER_OBJS) $(LIBS)


all: subdirs facialattributes


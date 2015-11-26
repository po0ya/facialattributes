USER_OBJS := histogram.o lbp.o

LIBS = $(shell pkg-config opencv --libs)
lbp: $(USER_OBJS)
	g++ -c $(USER_OBJS) $(LIBS)


all: lbp


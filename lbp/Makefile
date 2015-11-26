USER_OBJS := histogram.o lbp.o

LIBS := `pkg-config opencv --libs`
lbp: $(USER_OBJS)
	g++ -o "lbp" $(USER_OBJS) $(LIBS)


all: lbp


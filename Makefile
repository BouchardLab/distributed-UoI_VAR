TARGETS=uoi_var
OBJS=uoi_var.o bins.o matrix-operations.o lasso.o var-distribute-data.o var_kron.o

CXX=CC
CXXFLAGS=-Wall -g  -O3 -qopenmp -mkl -std=c++11  -fp-model fast=2 -xMIC-AVX512 $(EIGEN3)
CC=CC 
CCFLAGS=-g  -O3 -qopenmp -xMIC-AVX512

.PHONY: all clean

all: $(TARGETS)

uoi_var : $(OBJS)
	$(CXX)  -o $@ $(OBJS)  $(CXXFLAGS)

%.o : %.cpp
	$(CXX) -c $<  $(CXXFLAGS)

%.o : %.c
	$(CC) -c  $< $(CCFLAGS)

clean:
	rm -f $(TARGETS) $(OBJS)

MPI=mpicxx
SRC=main.cc matmul.cc
FLAGS=-O3 -fno-asynchronous-unwind-tables -mavx2 -mavx -mfma -ffast-math -funsafe-math-optimizations -floop-parallelize-all -ftree-parallelize-loops=4 -fprefetch-loop-arrays -fno-tree-vectorize -fprofile-use -w

EXE=matmul

OBJ=${SRC:.cc=.o}

all: ${EXE}

${EXE}: ${OBJ}
	${MPI} ${FLAGS} -o $@ $^

%.o: %.cc
	${MPI} ${FLAGS} -c $^ -o $@ -Wall

.PHONY: clean
clean:
	rm -rf ${OBJ} ${EXE}
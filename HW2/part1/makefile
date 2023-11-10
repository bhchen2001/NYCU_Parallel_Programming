CC=g++
SRC=pi_pthread
PFLAG=-lpthread -march=native

all:$(SRC).out

$(SRC).out: $(SRC).cpp
	$(CC) $(SRC).cpp -o $(SRC).out $(PFLAG)

clean:
	rm -f *.out *.o profiling_result perf.*
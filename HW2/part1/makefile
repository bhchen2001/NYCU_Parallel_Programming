CC=g++
SRC=pi_pthread
PFLAG=-lpthread

all:
	# please refer the flag time/gprof/perf/run

run: $(SRC).out

$(SRC).out: $(SRC).cpp $(SRC).o
	$(CC) $(SRC).cpp -o $(SRC).out $(PFLAG)

rm:
	rm -f *.out *.o profiling_result perf.*
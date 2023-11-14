CC=g++
SRC=pi
FLAGS= -lpthread -march=native -mavx2 -ffast-math -O3

all:$(SRC).out

$(SRC).out: $(SRC).cpp
	$(CC) $(SRC).cpp -o $(SRC).out $(FLAGS)

clean:
	rm -f *.out *.o
mpicxx ./hello.cc -o mpi_hello
./scp_script.sh
mpirun -np 8 --hostfile hosts mpi_hello
mpirun -np 16 --hostfile hosts mpi_hello
# ParallelHashing
An implementation of a hash tree using CUDA and MPI
#### Testfile creation
- `base64 /dev/urandom | head -c <file size in bytes> > test_file.txt`
- Applies for both CUDA and MPI.
- Make sure test_file.txt is in the root directory for both implementations
#### MPI 
##### Compilation
- `mpicc parallel_hashing.c -lssl -lcrypto -o ph`
##### Execution
- `mpirun -np <number of processes> ./ph`
- Number of processes should be a power of 2

#### CUDA
- Import the project into nsight
- Build the project
- Execute as C/C++ application (remotely or locally)
- Execution has been tested locally on a GPU (NVIDIA 920M 1.7G)







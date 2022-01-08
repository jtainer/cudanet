nvcc -c cudanet.cu matmul.cu
g++ -c test.cpp
g++ -L/usr/local/cuda/lib64 test.o cudanet.o matmul.o -lcudart


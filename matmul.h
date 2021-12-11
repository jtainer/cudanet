#include <cuda.h>

#ifndef MATMUL_H
#define MATMUL_H

__global__
void kernel(float* input, float* weights, float* output, const int numOfInputs, const int numOfNodes);



#endif

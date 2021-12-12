// 
// A kernel that multiplies two vectors and applies the sigmoid activation function
//
// 2021, Jonathan Tainer
//


#include <cuda.h>

#ifndef MATMUL_H
#define MATMUL_H

__global__
void forwardKernel(float* input, float* weights, float* output, const int numOfInputs, const int numOfNodes);



#endif

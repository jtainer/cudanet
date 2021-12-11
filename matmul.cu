// 
// A kernel that multiplies two vectors and applies the sigmoid activation function
// 
// 2021, Jonathan Tainer
// 

#include "matmul.h"

__global__
void kernel(float* input, float* weights, float* output, const int numOfInputs, const int numOfNodes) {
	int tid = threadIdx.x + (blockIdx * blockDim.x);

	float sum = 0;
	if (tid < numOfNodes) {
		for (int i = 0; i < numOfInputs; i++)
			sum += input[i] * weights[(i * numOfNodes) + tid];
		output[tid] = sum;
	}
}



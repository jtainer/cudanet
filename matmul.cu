// 
// A kernel that multiplies two vectors and applies the sigmoid activation function
// 
// 2021, Jonathan Tainer
// 

#include "matmul.h"

__global__
void forwardKernel(float* input, float* weights, float* output, const int numOfInputs, const int numOfNodes) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx * blockDim.x);

	// It is possible that the number of threads will exceed the number of nodes, so it is necessary to ensure that the excess threads do nothing
	if (tid < numOfNodes) {

		// Compute dot product of input vector and weight vector
		float sum = 0.f;
		for (int i = 0; i < numOfInputs; i++)
			sum += input[i] * weights[(i * numOfNodes) + tid];
		
		// Apply sigmoid activation function
		float result = 1.f / (1.f + exp(-1.f * sum));

		// Write result to output vector
		output[tid] = result;
	}
}

__global__
void backwardKernel(Layer layer

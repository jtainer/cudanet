// 
// Functions to create, train, and destroy a feed forward neural network on GPU
// 
// 2021, Jonathan Tainer
// 

#include "cudanet.h"
#include "matmul.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>i


void initLayer(Layer* sysLayer, int numOfInputs, int numOfNodes) {
	
	// Allocate memory for weight matrix
	sysLayer->weightMatrix = (float*)malloc(numOfInputs * numOfNodes * sizeof(float));

	// Initialize weight matrix with random values between -1.f and 1.f
	float multiplier = 2.f / RAND_MAX;
	for (int i = 0; i < numOfInputs * numOfNodes; i++)
		sysLayer->weightMatrix[i] = (rand() * multiplier) - 1.f;

	// Allocate memory for output vector
	sysLayer->outputVector = (float*)malloc((numOfNodes + 1) * sizeof(float));
	
	// Initialize the bias in the output vector to 1.f
	sysLayer->outputVector[numOfNodes] = 1.f; /* bias input */

	// Note the weight and node counts for this layer
	sysLayer->numOfNodes = numOfNodes;
	sysLayer->weightsPerNode = numOfInputs;
}

void deleteLayer(Layer* sysLayer) {

	// Deallocate memory for weight matrix and output vector
	free(sysLayer->weightMatrix);
	free(sysLayer->outputVector);
	
	// Point weight matrix and output vector to null
	sysLayer->weightMatrix = NULL;
	sysLayer->outputVector = NULL;
	
	// Note that the layer contains no weights or nodes
	sysLayer->numOfNodes = 0;
	sysLayer->weightsPerNode = 0;
}

// This function assumes that the destination layer does not have memory allocated to it
// Calling copyLayer on a destination layer which has memory allocated to it will result in a memory leak
void copyLayer(Layer* devLayer, Layer* sysLayer, int direction) {

	switch (direction) {

	case SYS_TO_DEV:

		// Copy node and weight count from system layer to device layer
		devLayer->numOfNodes = sysLayer->numOfNodes;
		devLayer->weightsPerNode = sysLayer->weightsPerNode;

		// Allocate an appropriate amount of video memory		
		cudaMalloc((void**)&devLayer->weightMatrix, sizeof(float) * devLayer->numOfNodes * devLayer->weightsPerNode);
		cudaMalloc((void**)&devLayer->outputVector, sizeof(float) * (devLayer->numOfNodes + 1));
		
		// Copy weight matrix and output vector from system memory to video memory
		cudaMemcpy(devLayer->weightMatrix, sysLayer->weightMatrix, sizeof(float) * devLayer->numOfNodes * devLayer->weightsPerNode, cudaMemcpyHostToDevice);
		cudaMemcpy(devLayer->outputVector, sysLayer->outputVector, sizeof(float) * (devLayer->numOfNodes + 1), cudaMemcpyHostToDevice);
		break;

	case DEV_TO_SYS:

		//Copy node and weight count from device layer to system layer
		sysLayer->numOfNodes = devLayer->numOfNodes;
		sysLayer->weightsPerNode = devLayer->weightsPerNode;
		
		// Allocate an appropriate amount of system memory
		sysLayer->weightMatrix = (float*)malloc(sysLayer->numOfNodes * sysLayer->weightsPerNode * sizeof(float));
		sysLayer->outputVector = (float*)malloc((sysLayer->numOfNodes + 1) * sizeof(float));

		// Copy weight matrix and output vector from video memory to system memory
		cudaMemcpy(sysLayer->weightMatrix, devLayer->weightMatrix, sizeof(float) * sysLayer->numOfNodes * sysLayer->weightsPerNode, cudaMemcpyDeviceToHost);
		cudaMemcpy(sysLayer->outputVector, devLayer->outputVector, sizeof(float) * (sysLayer->numOfNodes + 1), cudaMemcpyDeviceToHost);
		break;
	}
}

void cudaDeleteLayer(Layer* devLayer) {
	
	// Deallocate video memory for weight matrix and output vector
	cudaFree(devLayer->weightMatrix);
	cudaFree(devLayer->outputVector);
	
	// Point weight matrix and output vector to null
	devLayer->weightMatrix = NULL;
	devLayer->outputVector = NULL;
	
	// Note that the layer contains no weights or nodes
	devLayer->numOfNodes = 0;
	devLayer->weightsPerNode = 0;
}

// devLayer must point to an array of layers in video memory, not system memory
// inputVector and outputVector must point to system memory
void forwardPass(Layer* devLayer, int numOfLayers, float* inputVector, float* outputVector) {
	
	// Copy input vector to video memory
	float* devInputVector;
	cudaMalloc((void**)&devInputVector, sizeof(float) * devLayer[0].weightsPerNode);
	cudaMemcpy(devInputVector, inputVector, sizeof(float) * devLayer[0].weightsPerNode, cudaMemcpyHostToDevice);

	// Run the kernel for the first layer, using the input vector provided by the function call
	forwardKernel<<<(layer[0].numOfNodes / 256) + 1, 256>>>
		(devInputVector, devLayer[0].weightMatrix, devLayer[0].outputVector, devLayer[0].weightsPerNode, devLayer[0].numOfNodes);

	// Deallocate video memory for input vector
	cudaFree(devInputVector);

	// Iterate through the layers of the network starting with the second layer
	// Run the kernel for each iteration, using the output of the previous layer as the input for the current layer
	for (int n = 1; n < numOfLayers; n++) {
		forwardKernel<<<(layer[n].numOfNodes / 256) + 1, 256>>>
			(devLayer[n - 1], devLayer[n].weightMatrix, devLayer[n].outputVector, devLayer[n].weightsPerNode, devLayer[n].numOfNodes);
	}

	// Copy final output vector to system memory
	cudaMemcpy(outputVector, devLayer[numOfLayers - 1].outputVector, sizeof(float) * devLayer[numOfLayers - 1].numOfNodes, cudaMemcpyDeviceToHost);

}



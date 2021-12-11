// 
// Functions to create, train, and destroy a feed forward neural network on GPU
// 
// 2021, Jonathan Tainer
// 

#include "cudanet.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>i


void initLayer(Layer* sysLayer, int numOfInputs, int numOfNodes) {
	if (numOfInputs > 0)
		sysLayer->weightMatrix = (float*)malloc(numOfInputs * numOfNodes * sizeof(float));
	sysLayer->outputVector = (float*)malloc((numOfNodes + 1) * sizeof(float));
	sysLayer->outputVector[numOfNodes] = 1.f; /* bias input */

	sysLayer->numOfNodes = numOfNodes;
	sysLayer->weightsPerNode = numOfInputs;
}

void deleteLayer(Layer* sysLayer) {
	free(sysLayer->weightMatrix);
	free(sysLayer->outputVector);
	sysLayer->weightMatrix = NULL;
	sysLayer->outputVector = NULL;
	sysLayer->numOfNodes = 0;
	sysLayer->weightsPerNode = 0;
}

void copyLayer(Layer* devLayer, Layer* sysLayer, int direction) {
	switch (direction) {

	case SYS_TO_DEV:
		devLayer->numOfNodes = sysLayer->numOfNodes;
		devLayer->weightsPerNode = sysLayer->weightsPerNode;
		
		cudaMalloc((void**)&devLayer->weightMatrix, sizeof(float) * devLayer->numOfNodes * devLayer->weightsPerNode);
		cudaMalloc((void**)&devLayer->outputVector, sizeof(float) * (devLayer->numOfNodes + 1));
		
		cudaMemcpy(devLayer->weightMatrix, sysLayer->weightMatrix, sizeof(float) * devLayer->numOfNodes * devLayer->weightsPerNode, cudaMemcpyHostToDevice);
		cudaMemcpy(devLayer->outputVector, sysLayer->outputVector, sizeof(float) * (devLayer->numOfNodes + 1), cudaMemcpyHostToDevice);
		break;

	case DEV_TO_SYS:
		sysLayer->numOfNodes = devLayer->numOfNodes;
		sysLayer->weightsPerNode = devLayer->weightsPerNode;
		
		sysLayer->weightMatrix = (float*)malloc(sysLayer->numOfNodes * sysLayer->weightsPerNode * sizeof(float));
		sysLayer->outputVector = (float*)malloc((sysLayer->numOfNodes + 1) * sizeof(float));

		cudaMemcpy(sysLayer->weightMatrix, devLayer->weightMatrix, sizeof(float) * sysLayer->numOfNodes * sysLayer->weightsPerNode, cudaMemcpyDeviceToHost);
		cudaMemcpy(sysLayer->outputVector, devLayer->outputVector, sizeof(float) * (sysLayer->numOfNodes + 1), cudaMemcpyDeviceToHost);
		break;
	}
}

void cudaDeleteLayer(Layer* devLayer) {
	cudaFree(devLayer->weightMatrix);
	cudaFree(devLayer->outputVector);
	devLayer->weightMatrix = NULL;
	devLayer->outputVector = NULL;
	devLayer->numOfNodes = 0;
	devLayer->weightsPerNode = 0;
}



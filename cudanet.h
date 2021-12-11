// 
// Functions to create, train, and destroy a feedforward neural network on GPU
// 
// 2021, Jonathan Tainer
//	

#ifndef SYS_TO_DEV
#define SYS_TO_DEV 0
#endif

#ifndef DEV_TO_SYS
#define DEV_TO_SYS 1
#endif

#ifndef CUDANET_H
#define CUDANET_H

typedef struct Layer {
	float* weightMatrix;
	float* outputVector;
	int numOfNodes;
	int weightsPerNode;
} Layer;

/* Functions for creating, copying, and destroying layers in system and device memory */

void initLayer(Layer* sysLayer, int numOfInputs, int numOfNodes);

void deleteLayer(Layer* sysLayer);

void copyLayer(Layer* devLayer, Layer* sysLayer, int direction);

void cudaDeleteLayer(Layer* devLayer);

/* Forward propagation functions */

#endif

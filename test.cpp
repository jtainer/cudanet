#include "cudanet.h"
#include <stdio.h>

int main() {
	Network net = createNetwork(4, 4, 4, 4);

	deleteNetwork(&net);

	return 0;
}

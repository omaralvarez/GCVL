// CUDA-C includes
#include "test.h"
#include "../cudautils.h"

#include <cstdio>

__global__ void addAry( int * ary1, int * ary2 )
{
    int indx = threadIdx.x;
    ary1[ indx ] += ary2[ indx ];
}


// Main cuda function

void runCudaPart() {

    int * ary1 = new int[32];
    int * ary2 = new int[32];
    int * res = new int[32];

    for( int i=0 ; i<32 ; i++ )
    {
        ary1[i] = i;
        ary2[i] = 2*i;
        res[i]=0;
    }

	CUDA_Array<int> d_ary1, d_ary2;
	d_ary1.Initialize(32,ary1);

    int * d_ary1, *d_ary2;
    cudaMalloc((void**)&d_ary1, 32*sizeof(int));
    cudaMalloc((void**)&d_ary2, 32*sizeof(int));


    cudaMemcpy((void*)d_ary1, (void*)ary1, 32*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_ary2, (void*)ary2, 32*sizeof(int), cudaMemcpyHostToDevice);


    addAry<<<1,32>>>(d_ary1,d_ary2);

    cudaMemcpy((void*)res, (void*)d_ary1, 32*sizeof(int), cudaMemcpyDeviceToHost);
    for( int i=0 ; i<32 ; i++ )
        printf( "result[%d] = %d\n", i, res[i]);


    cudaFree(d_ary1);
    cudaFree(d_ary2);
}

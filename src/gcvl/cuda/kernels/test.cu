// CUDA-C includes
#include "test.cuh"
#include "../cudautils.h"

#include <cstdio>

__global__ void addAry( int * ary1, int * ary2, int * res)
{
    int indx = threadIdx.x;
    res[ indx ] = ary1[ indx ] + ary2[ indx ];
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

	CUDA_Array<int> d_ary1, d_ary2, d_res;
	d_ary1.Initialize(32,ary1);
	d_ary2.Initialize(32,ary2);
	d_res.Initialize(32, res);

    d_ary1.Host_to_Device();
	d_ary2.Host_to_Device();

    addAry<<<1,32>>>(d_ary1.Get_Device_Array(),d_ary2.Get_Device_Array(), d_res.Get_Device_Array());

    d_res.Device_to_Host();
    for( int i=0 ; i<32 ; i++ )
        printf( "result[%d] = %d\n", i, res[i]);


    d_ary1.Release_Memory();
	d_ary2.Release_Memory();
	d_res.Release_Memory();
	delete [] ary1;
	delete [] ary2;
	delete [] res;

}

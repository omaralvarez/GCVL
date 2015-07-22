/*
 *
 * GPGPU Computer Vision Library (GCVL)
 *
 * Copyright (c) Luis Omar Alvarez Mures 2015 <omar.alvarez@udc.es>
 * Copyright (c) Emilio Padron Gonzalez 2015 <emilioj@gmail.com>
 *
 * All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library.
 *
 */

#include "blockmatching.cuh"
#include "../cudautils.h"

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_BLK 8

__global__ void addAry( int * ary1, int * ary2, int * res)
{
    int indx = threadIdx.x;
    res[ indx ] = ary1[ indx ] + ary2[ indx ];
}

__global__ void calculateDisparity(const unsigned char * inputLeft, const unsigned char * inputRight,
								   unsigned char * output, const int width, const int height, const int dim,
								   const int radius, const int maxDisp)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int offsetx = x - radius;
	const int offsety = y - radius;

	if (offsetx >= 0 && offsetx + dim < width && offsety >= 0 && offsety + dim < height) {

		unsigned int sum = 0;
		unsigned int bestSum = -1;
		unsigned int bestd = 0;

		for (int d = 0; d < maxDisp; ++d) {
			for (int i = offsety; i < dim + offsety; ++i) {
				for (int j = offsetx; j < dim + offsetx; ++j) {
					if (j - d >= 0)
						sum += abs((int)inputLeft[i * width + j] - (int)inputRight[i * width + j - d]);
					else
						sum += abs((int)inputLeft[i * width + j]);
				}
			}
			if (sum < bestSum) {
				bestSum = sum;
				bestd = d;
			}
			sum = 0;
		}

		output[y * width + x] = bestd;

	}

}

__global__ void normalizeMap(unsigned char * input, unsigned char *output, const unsigned int width, const int maxDisp) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	output[y * width + x] = (input[y * width + x] / (float)maxDisp) * 255;

}

void launchBM(const unsigned char * inputLeft, const unsigned char * inputRight,
			unsigned char * output, const int width, const int height, const int dim,
			const int radius, const int maxDisp) 
{

	dim3 dimBlock(CUDA_BLK, CUDA_BLK);
	dim3 gridSize((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
	calculateDisparity<<<gridSize, dimBlock>>>(inputLeft, inputRight, output, width, height, dim, radius, maxDisp);

}

void launchNormalization(unsigned char * input, unsigned char * output, const int width, const int height, const int maxDisp)
{

	dim3 dimBlock(CUDA_BLK, CUDA_BLK);
	dim3 gridSize((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
	normalizeMap<<<gridSize, dimBlock>>>(input, output, width, maxDisp);

}


// Main cuda function

/*void runCudaPart() {

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

}*/

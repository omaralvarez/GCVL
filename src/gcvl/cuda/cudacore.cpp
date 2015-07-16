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

#include "cudacore.h"
#include "cudautils.h"

#include <iostream>
#include <stdio.h> 
#include <assert.h> 
#include <cuda_runtime.h> 
//#include "helper_cuda.h"
#include "kernels/test.h"

using namespace gcvl::cuda;

Core::Core() {
    
	std::cout << " **** Initializing CUDA Core ****" << std::endl;

	CUDA_devices_list list;
	list.Print();
	
	/*int deviceCount;
	    cudaGetDeviceCount(&deviceCount);
	    if (deviceCount == 0)
	        printf("There is no device supporting CUDA\n");
	    int dev;
	    for (dev = 0; dev < deviceCount; ++dev) {
	        cudaDeviceProp deviceProp;
	        cudaGetDeviceProperties(&deviceProp, dev);
	        if (dev == 0) {
	            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
               printf("There is no device supporting CUDA.\n");
	            else if (deviceCount == 1)
	                printf("There is 1 device supporting CUDA\n");
	            else
	                printf("There are %d devices supporting CUDA\n", deviceCount);
	        }
	        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	        printf("  Major revision number:                         %d\n",
	               deviceProp.major);
	        printf("  Minor revision number:                         %d\n",
	               deviceProp.minor);
	        printf("  Total amount of global memory:                 %u bytes\n",
	               deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
	        printf("  Number of multiprocessors:                     %d\n",
	               deviceProp.multiProcessorCount);
	        printf("  Number of cores:                               %d\n",
	               8 * deviceProp.multiProcessorCount);
	    #endif
	        printf("  Total amount of constant memory:               %u bytes\n",
	               deviceProp.totalConstMem);
	        printf("  Total amount of shared memory per block:       %u bytes\n",
               deviceProp.sharedMemPerBlock);
	        printf("  Total number of registers available per block: %d\n",
	               deviceProp.regsPerBlock);
	        printf("  Warp size:                                     %d\n",
	               deviceProp.warpSize);
	        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
	        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
	               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
	               deviceProp.maxThreadsDim[2]);
	        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
	               deviceProp.maxGridSize[0],
	               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
	        printf("  Maximum memory pitch:                          %u bytes\n",
	               deviceProp.memPitch);
	        printf("  Texture alignment:                             %u bytes\n",
	               deviceProp.textureAlignment);
	        printf("  Clock rate:                                    %.2f GHz\n",
	               deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
	        printf("  Concurrent copy and execution:                 %s\n",
	               deviceProp.deviceOverlap ? "Yes" : "No");
	    #endif
    }*/

		runCudaPart();

	    printf("\nTest PASSED\n");
    
}
	
Core::~Core() { 
	std::cout << " **** Destroying CUDA Core ****" << std::endl;
}



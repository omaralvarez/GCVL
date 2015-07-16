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

#pragma once

#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <vector>

#define CUDA_Test_Success(err, fct_name)                            \
if ((err) != cudaSuccess)                                           \
{                                                                   \
    std::cout                                                        \
        << "ERROR calling " << fct_name << "() ("                   \
        << __FILE__ << " line " << __LINE__ << "): "                \
        << CUDA_Error_to_String(err) << "\n" << std::flush;         \
    abort();                                                        \
}

#ifdef __DRIVER_TYPES_H__
static const std::string CUDA_Error_to_String(cudaError_t error);
#endif
#ifdef __cuda_cuda_h__
static const std::string CUDA_Error_to_String(CUresult error);
#endif

class CUDA_device;
class CUDA_devices_list;

class CUDA_device
{
private:
	int id;
	int asyncEngineCount;
	int canMapHostMemory;
	int clockRate;
	int computeMode;
	int concurrentKernels;
	int deviceOverlap;
	int ECCEnabled;
	int integrated;
	int kernelExecTimeoutEnabled;
	int l2CacheSize;
	int major;
	int maxGridSize[3];
	int maxSurface1D;
	int maxSurface1DLayered[2];
	int maxSurface2D [2];
	int maxSurface2DLayered[3];
	int maxSurface3D[3];
	int maxSurfaceCubemap;
	int maxSurfaceCubemapLayered[2];
	int maxTexture1D;
	int maxTexture1DLayered[2];
	int maxTexture1DLinear;
	int maxTexture2D[2];
	int maxTexture2DGather[2];
	int maxTexture2DLayered[3];
	int maxTexture2DLinear[3];
	int maxTexture3D[3];
	int maxTextureCubemap;
	int maxTextureCubemapLayered[2];
	int maxThreadsDim[3];
	int maxThreadsPerBlock;
	int maxThreadsPerMultiProcessor;
	int memoryBusWidth;
	int memoryClockRate;
	size_t memPitch;
	int minor;
	int multiProcessorCount;
	std::string name;
	int pciBusID;
	int pciDeviceID;
	int pciDomainID;
	int regsPerBlock;
	size_t sharedMemPerBlock;
	size_t surfaceAlignment;
	int tccDriver;
	size_t textureAlignment;
	size_t texturePitchAlignment;
	size_t totalConstMem;
	size_t totalGlobalMem;
	int unifiedAddressing;
	int warpSize;

public:
	CUDA_device(int id = 0);
	//~CUDA_device();
	void Set_Information(const int _id);
	void Print() const;
};

class CUDA_devices_list
{
private:
	int count;
	std::vector<CUDA_device> device_list;

public:
	CUDA_devices_list();
	//~CUDA_devices_list();
	void Print();
};

template <class T>
class CUDA_Array
{
private:
    int N;                              // Number of elements in array
    size_t sizeof_element;              // Size of each array elements
    size_t new_array_size_bytes;      // Size (bytes) of new padded array
    T * host_array;                  // Pointer to start of host array
	T * device_array;				// Pointer to device array
	cudaError_t err; 

public:
    CUDA_Array();
    void Initialize(int _N, T *&host_array);
    void Release_Memory();
    void Host_to_Device();
    void Device_to_Host();
    //void Validate_Data();

    inline T * Get_Device_Array() { return device_array; }
    inline T * Get_Host_Pointer() { return  host_array;   }
    //void Set_as_Kernel_Argument(cl_kernel &kernel, const int order);
};

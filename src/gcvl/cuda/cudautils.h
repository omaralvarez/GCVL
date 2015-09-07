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

// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
typedef struct
{
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int cores;
} sSMtoCores;

const static sSMtoCores nGpuArchCoresPerSM[] =
{
    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    {   -1, -1 }
};

//! Obtains the estimated number of cores per SM.
/*!
  \param major major architecture version number.
  \param minor minor architecture version number.
  \return number of compute units per SM.
*/
inline int SMVer2CU(int major, int minor) {

	int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {

        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }

        index++;

    }

    // If we don't find the values, we default to using the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].cores);
    return nGpuArchCoresPerSM[index-1].cores;

}

class CUDA_device;
class CUDA_devices_list;

class CUDA_device
{
private:
	int id;
	cudaDeviceProp properties;
	bool compute;
	int sm_per_multiproc;
	int compute_units;
	unsigned long long compute_perf;
	cudaError_t err;

public:
	CUDA_device(int id = -1);
	//~CUDA_device();
	bool operator<(const CUDA_device &b) const;
	void Set_Information(unsigned long long _compute_perf, int arch);
	void Print() const;
  //! Obtains the GPU device name.
  /*!
    \return device name.
  */
	inline std::string Get_Name() const { return properties.name; }
  //! Obtains the device id.
  /*!
    \return device id.
  */
	inline int Get_ID() const { return id; }
};

class CUDA_devices_list
{
private:
	int count;
	std::vector<CUDA_device> device_list;
	bool is_initialized;
	int preferred_device;
	cudaError_t err;
public:
	CUDA_devices_list();
	//~CUDA_devices_list();
	void Initialize();
  //! Sets the preferred CUDA device.
  /*!
    \param _preferred_device device id that is going to be used, -1 if we want GCVL to guess which is the best device available.
  */
	inline void Set_Preferred_CUDA(const int _preferred_device = -1) { preferred_device = _preferred_device; }
  //! Obtains the preferred CUDA device id.
  /*!
    \return device id.
  */
	inline int Preferred_CUDA() { return preferred_device; }
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

    //! Obtains the device array CUDA memory pointer.
    /*!
      \return CUDA device memory pointer.
    */
    inline T * Get_Device_Array() { return device_array; }
    //! Obtains the host array memory pointer.
    /*!
      \return memory pointer.
    */
    inline T * Get_Host_Pointer() { return  host_array; }

};

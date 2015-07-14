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

template <class T>
class CUDA_Array
{
private:
    int N;                              // Number of elements in array
    size_t sizeof_element;              // Size of each array elements
    uint64_t new_array_size_bytes;      // Size (bytes) of new padded array
    T     *host_array;                  // Pointer to start of host array, INCLUDIǸG padding
    //std::string platform;               // OpenCL platform
    //cl_context context;                 // OpenCL context
    //cl_command_queue command_queue;     // OpenCL command queue
    //cl_device_id device;                // OpenCL device
    //cl_int err;                         // Error code

    // Allocated memory on device
    //cl_mem device_array;                // Memory of device
    //cl_mem cl_array_size_bit;
    //cl_mem cl_sha512sum;

public:
    CUDA_Array();
    void Initialize(int _N, const size_t _sizeof_element,
                    T *&host_array);
    void Release_Memory();
    void Host_to_Device();
    void Device_to_Host();
    //void Validate_Data();

    //inline cl_mem * Get_Device_Array() { return &device_array; }
    inline T *      Get_Host_Pointer() { return  host_array;   }
    //void Set_as_Kernel_Argument(cl_kernel &kernel, const int order);
};

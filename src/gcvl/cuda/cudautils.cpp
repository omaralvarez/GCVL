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

#include "cudautils.h"

#include <cassert>

template <class T>
CUDA_Array<T>::CUDA_Array()
{
    N                           = 0;
    sizeof_element              = 0;
    new_array_size_bytes        = 0;
    host_array                  = NULL;
	device_array                = NULL;
}

template <class T>
void CUDA_Array<T>::Initialize(int _N, T *&_host_array)
{
    assert(_host_array != NULL);

    N               = _N;
    sizeof_element  = sizeof(T);
    host_array      = _host_array;
    new_array_size_bytes = N * sizeof_element;

	err = cudaMalloc(&device_array, new_array_size_bytes);

	//TODO check error
    //OpenCL_Test_Success(err, "clCreateBuffer()");
}

template <class T>
void CUDA_Array<T>::Release_Memory()
{
    if (device_array)
        cudaFree(device_array);
}

template <class T>
void CUDA_Array<T>::Host_to_Device()
{
	err = cudaMemcpy(device_array, host_array, new_array_size_bytes, cudaMemcpyHostToDevice);
	//TODO check errors
}

// *****************************************************************************
template <class T>
void CUDA_Array<T>::Device_to_Host()
{
	err = cudaMemcpy(host_array, device_array, new_array_size_bytes, cudaMemcpyDeviceToHost);
	//TODO check errors
}

//TODO Add more types?
template class CUDA_Array<float>;
template class CUDA_Array<double>;
template class CUDA_Array<int>;
template class CUDA_Array<unsigned int>;
template class CUDA_Array<char>;
template class CUDA_Array<unsigned char>;

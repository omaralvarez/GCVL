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

template <class T>
CUDA_Array<T>::CUDA_Array()
{
    N                           = 0;
    sizeof_element              = 0;
    new_array_size_bytes        = 0;
    host_array                  = NULL;
}

template <class T>
void CUDA_Array<T>::Initialize(int _N, const size_t _sizeof_element,
                                 T *&_host_array)
{
    assert(_host_array != NULL);

    N               = _N;
    sizeof_element  = _sizeof_element;
    host_array      = _host_array;
    new_array_size_bytes = N * sizeof_element;

    //device_array = clCreateBuffer(context, flags, new_array_size_bytes, NULL, &err);
    //OpenCL_Test_Success(err, "clCreateBuffer()");
}

template <class T>
void CUDA_Array<T>::Release_Memory()
{
    //if (device_array)
        //clReleaseMemObject(device_array);
}

template <class T>
void CUDA_Array<T>::Host_to_Device()
{

}

// *****************************************************************************
template <class T>
void CUDA_Array<T>::Device_to_Host()
{

}

//TODO Add more types?
template class CUDA_Array<float>;
template class CUDA_Array<double>;
template class CUDA_Array<int>;
template class CUDA_Array<unsigned int>;
template class CUDA_Array<char>;
template class CUDA_Array<unsigned char>;

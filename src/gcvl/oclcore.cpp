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

#include "oclcore.h"

#include <iostream>

using namespace gcvl::opencl;

Core::Core() {
    
	std::cout << " **** Initializing OpenCL Core ****" << std::endl;

    platforms.Initialize("-1");

    // By passing "-1", to Initialize(), the first platform in the list
    // is chosen. Get that platform's string for later reference.
    platform = platforms.Get_Running_Platform();

	context = platforms[platform].Preferred_OpenCL_Device_Context();

	device = platforms[platform].Preferred_OpenCL_Device();
    
    // Lock the best device available on the platform. Devices are
    // ordered by number of max compute units (CL_DEVICE_MAX_COMPUTE_UNITS).
    // GPUs normally have at least a couple dozen compute units. On
    // CPUs, CL_DEVICE_MAX_COMPUTE_UNITS is the number of cores.
    // The locking is done by creating the file /tmp/gpu_usage.txt
    // where the platform and device is saved.
    platforms[platform].Lock_Best_Device();
    
    // Print All information possible on the platforms and their devices.
    platforms.Print();
    
    // Create a command queue on "platform"'s preferred device.
    cl_int err;
	queue = clCreateCommandQueue(	context,  // OpenCL context
									device,          // OpenCL device id
									0, &err);
    
}

Core::Core(std::string p, bool locking) {
    
	std::cout << " **** Initializing OpenCL Core ****" << std::endl;

    platforms.Initialize(p, locking);

    platform = p;

	context = platforms[platform].Preferred_OpenCL_Device_Context();

	device = platforms[platform].Preferred_OpenCL_Device();
    
    platforms[platform].Lock_Best_Device();

    cl_int err;
	queue = clCreateCommandQueue(	context,  // OpenCL context
									device,          // OpenCL device id
									0, &err);
    
}
	
Core::~Core() { 
	std::cout << " **** Destroying OpenCL Core ****" << std::endl;
}



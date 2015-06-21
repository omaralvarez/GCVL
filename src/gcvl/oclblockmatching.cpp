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

#include "oclblockmatching.h"
#include "oclutils.h"
#include <iostream>

using namespace gcvl::opencl;

BlockMatching::BlockMatching(Core * core, unsigned int n, float * input, float * output) {
    
	std::cout << " **** Initializing OpenCL BlockMatching ****" << std::endl;
    
    _core = core;
    _kernel.Initialize("../../src/gcvl/kernels/test.cl", _core->getContext(), _core->getDevice());
    _n = n;
    _input = input;
    _output = output;
    cl_context context = _core->getContext();
    cl_command_queue queue = _core->getQueue();
    cl_device_id device = _core->getDevice();
    _clInput.Initialize(_n, sizeof(float), _input, context, CL_MEM_READ_ONLY, _core->getPlatform(), queue, device, false);
    _clInput.Host_to_Device();
    _clOutput.Initialize(_n, sizeof(float), _output, context, CL_MEM_WRITE_ONLY, _core->getPlatform(), queue, device, false);
    
}
	
BlockMatching::~BlockMatching() { 
	std::cout << " **** Destroying OpenCL BlockMatching ****" << std::endl;
}

void BlockMatching::prepare() {
    
	std::cout << " **** prepare OpenCL BlockMatching ****" << std::endl;
    
    _kernel.Build("add");
    _kernel.Compute_Work_Size(_n, 1, 32, 1);
    
}

void BlockMatching::setArgs() {
    
	std::cout << " **** setArgs OpenCL BlockMatching ****" << std::endl;
    
    cl_kernel kernel = _kernel.Get_Kernel();
    _clInput.Set_as_Kernel_Argument(kernel, 0);
    _clOutput.Set_as_Kernel_Argument(kernel, 1);
    
    
}

void BlockMatching::launch() {
    
	std::cout << " **** launch OpenCL BlockMatching ****" << std::endl;
    
    _kernel.Launch(_core->getQueue());
    
}

void BlockMatching::postpare() {
    
	std::cout << " **** postpare OpenCL BlockMatching ****" << std::endl;
    
    _clOutput.Device_to_Host();
    
}



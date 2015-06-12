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
#include <iostream>

using namespace gcvl::opencl;

BlockMatching::BlockMatching() { 
	std::cout << " **** Initializing OpenCL BlockMatching ****" << std::endl;
}
	
BlockMatching::~BlockMatching() { 
	std::cout << " **** Destroying OpenCL BlockMatching ****" << std::endl;
}

void BlockMatching::prepare() {
	std::cout << " **** prepare OpenCL BlockMatching ****" << std::endl;
}
void BlockMatching::setArgs() {
	std::cout << " **** setArgs OpenCL BlockMatching ****" << std::endl;
}
void BlockMatching::launch() {
	std::cout << " **** launch OpenCL BlockMatching ****" << std::endl;
}
void BlockMatching::postpare() {
	std::cout << " **** postpare OpenCL BlockMatching ****" << std::endl;
}



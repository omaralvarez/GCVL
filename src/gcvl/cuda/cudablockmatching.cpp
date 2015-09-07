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

#include "cudablockmatching.h"
#include "cudautils.h"
#include "kernels/blockmatching.cuh"
#include <iostream>

using namespace gcvl::cuda;

//! Class constructor.
/*!
  \param core OpenCL core instance in which the user wants to run the Block Matching algorithm.
  \param inputLeft path to the left input image.
	\param inputRight path to the right input image.
	\param output pointer to the Block Matching resulting disparity map.
  \sa ~BlockMatching()
*/
BlockMatching::BlockMatching(Core & core, std::string inputLeft, std::string inputRight, std::unique_ptr<unsigned char[]> &output) {

	std::cout << " **** Initializing CUDA BlockMatching ****" << std::endl;

    _core = &core;
    _dim = 9;
    _radius = 4;
    _maxDisp = 255;
    _normalize = false;
    _inputLeft = cv::imread(inputLeft, CV_LOAD_IMAGE_GRAYSCALE);
    _inputRight = cv::imread(inputRight, CV_LOAD_IMAGE_GRAYSCALE);

    std::cout << "File: " << inputLeft << " Image size: " << _inputLeft.rows << "x" << _inputLeft.cols << std::endl;

    _width = _inputLeft.cols;
    _height = _inputLeft.rows;
    //output = new unsigned char[_width*_height];
	output.reset(new unsigned char[_width*_height]);
    _output = output.get();
    _cuInputLeft.Initialize(_width*_height, _inputLeft.data);
    _cuInputLeft.Host_to_Device();
    _cuInputRight.Initialize(_width*_height, _inputRight.data);
    _cuInputRight.Host_to_Device();
    _cuOutput.Initialize(_width*_height, _output);

}

//! Class destructor.
/*!
  \sa BlockMatching()
*/
BlockMatching::~BlockMatching() {

	std::cout << " **** Destroying CUDA BlockMatching ****" << std::endl;

    _cuInputLeft.Release_Memory();
    _cuInputRight.Release_Memory();
    _cuOutput.Release_Memory();

}

//! Function that performs pre-processing steps.
void BlockMatching::prepare() {

	std::cout << " **** prepare CUDA BlockMatching ****" << std::endl;

}

//TODO Remove, in CUDA not needed.
//! Function that sets arguments.
void BlockMatching::setArgs() {

	std::cout << " **** setArgs CUDA BlockMatching ****" << std::endl;

}

//! Launch the CUDA kernels.
void BlockMatching::launch() {

	std::cout << " **** launch CUDA BlockMatching ****" << std::endl;

	/*for (size_t i = 0; i < 20; i++)
	{
		std::cout << (unsigned int)_cuInputLeft.Get_Host_Pointer()[i] << " " << (unsigned int)_output[i] << std::endl;
	}*/

	launchBM(_cuInputLeft.Get_Device_Array(), _cuInputRight.Get_Device_Array(), _cuOutput.Get_Device_Array(),
		     _width, _height, _dim, _radius, _maxDisp);

    if (_normalize) {
		launchNormalization(_cuOutput.Get_Device_Array(), _cuOutput.Get_Device_Array(), _width, _height, _maxDisp);
    }

}

//! Function that waits for the kernels to finish processing.
void BlockMatching::postpare() {

	std::cout << " **** postpare CUDA BlockMatching ****" << std::endl;

    _cuOutput.Device_to_Host();
	/*for (size_t i = 0; i < 20; i++)
	{
		std::cout << (unsigned int)_cuInputLeft.Get_Host_Pointer()[i] << " " << (unsigned int)_output[i] << std::endl;
	}*/
    //_core->waitForQueue();

}

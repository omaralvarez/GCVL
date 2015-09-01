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

#include "blockmatching.h"
#include <iostream>

using namespace gcvl;

//! Class constructor.
/*!
  \param inputLeft path to the left input image.
	\param inputRight path to the right input image.
	\param output pointer to the Block Matching resulting disparity map.
  \sa ~BlockMatching()
*/
BlockMatching::BlockMatching(std::string inputLeft, std::string inputRight, std::unique_ptr<unsigned char[]> &output) {

	std::cout << " **** Initializing BlockMatching ****" << std::endl;

    _dim = 9;
    _radius = 4;
    _maxDisp = 255;
    _normalize = false;
    _inputLeft = cv::imread(inputLeft, CV_LOAD_IMAGE_GRAYSCALE);
    _inputRight = cv::imread(inputRight, CV_LOAD_IMAGE_GRAYSCALE);

    std::cout << "File: " << inputLeft << " Image size: " << _inputLeft.rows << "x" << _inputLeft.cols << std::endl;

    _width = _inputLeft.cols;
    _height = _inputLeft.rows;
	output.reset(new unsigned char[_width*_height]);
    _output = output.get();

}

//! Class destructor.
/*!
  \sa BlockMatching()
*/
BlockMatching::~BlockMatching() {

	std::cout << " **** Destroying BlockMatching ****" << std::endl;

}

//! Function that performs pre-processing steps.
void BlockMatching::prepare() {

	std::cout << " **** prepare BlockMatching ****" << std::endl;

}

//! Function that sets the algorithm arguments.
void BlockMatching::setArgs() {

	std::cout << " **** setArgs BlockMatching ****" << std::endl;

}

//! Launch the algorithm.
void BlockMatching::launch() {

	std::cout << " **** launch BlockMatching ****" << std::endl;

	#pragma omp parallel for
    for (int x = 0; x < _width; ++x) {
        for (int y = 0; y < _height; ++y) {

            const int offsetx = x - _radius;
            const int offsety = y - _radius;

            if(offsetx >= 0 && offsetx + _dim < _width && offsety >= 0 && offsety + _dim < _height) {

                unsigned int sum = 0;
                unsigned int bestSum = -1;
                unsigned int bestd = 0;

                for(int d = 0; d < _maxDisp; ++d) {
                    for(int i = offsety; i < _dim + offsety; ++i) {
                        for(int j = offsetx; j < _dim + offsetx; ++j) {
                            if(j - d >= 0)
                                sum += abs((int)_inputLeft.data[i * _width + j] - (int)_inputRight.data[i * _width + j - d]);
                            else
                                sum += abs((int)_inputLeft.data[i * _width + j]);
                        }
                    }
                    if(sum < bestSum) {
                        bestSum = sum;
                        bestd = d;
                    }
                    sum = 0;
                }

                _output[y * _width + x] = bestd;

            }

        }
    }

}

//! Function that performs optional normalization.
void BlockMatching::postpare() {

	std::cout << " **** postpare BlockMatching ****" << std::endl;

    if(_normalize)
		#pragma omp parallel for
        for (int x = 0; x < _width; ++x)
            for (int y = 0; y < _height; ++y)
                _output[y * _width + x] = (_output[y * _width + x]/(float)_maxDisp)*255;

}

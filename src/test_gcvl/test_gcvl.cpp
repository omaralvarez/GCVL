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

#include <gcvl/oclcore.h>
#include <gcvl/oclblockmatching.h>
#include <opencv2/opencv.hpp>

int main() {

    /*unsigned int n = 64;
    float * input = new float[n];
    float * output = new float[n];
    
    for(unsigned int i = 0; i < n; ++i) {
        input[i] = output[i] = 0.f;
    }*/
    
    cv::Mat image;
    image = cv::imread( "../data/tsukuba_r.png" );
    
    //unsigned char * input = new unsigned char[image.rows*image.cols*3];
    unsigned char * output = new unsigned char[image.rows*image.cols*3];
    
	gcvl::opencl::Core core;
	//gcvl::opencl::BlockMatching bm(&core, n, input, output);
    gcvl::opencl::BlockMatching bm(&core, image.rows, image.cols, image.data, output);
    bm.compute();
    
    cv::Mat out(image.rows, image.cols, CV_8UC3, output);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", out );
    //cv::waitKey(0);
    
    /*for(unsigned int i = 0; i < n; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    for(unsigned int i = 0; i < n; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;*/

	return 0;
}
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

#include <gcvl/opencl/oclcore.h>
#include <gcvl/opencl/oclblockmatching.h>
#include <gcvl/blockmatching.h>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        std::cout << "Usage: test_gcvl path/to/image_left path/to/image_right" << std::endl;
        return 0;
    }
    
	int dim = 5, maxDisp = 16;
	bool norm = true;
	std::unique_ptr<unsigned char[]> output;
    
    gcvl::BlockMatching bmCPU(argv[1], argv[2], output);
    bmCPU.setAggDim(dim);
	bmCPU.setMaxDisp(maxDisp);
	bmCPU.setNormalize(norm);
    bmCPU.compute();
    
	gcvl::opencl::Core core;
    gcvl::opencl::BlockMatching bmOCL(&core, argv[1], argv[2], output);
    bmOCL.setAggDim(dim);
	bmOCL.setMaxDisp(maxDisp);
	bmOCL.setNormalize(norm);
    bmOCL.compute();
    
    /*cv::Mat out(bmCPU.getHeight(), bmCPU.getWidth(), CV_8UC1, output.get());
    
    //cv::namedWindow( "Source Image", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //cv::imshow( "Source Image", image );
    cv::namedWindow( "Disparity Map", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Disparity Map", out );
    cv::waitKey(0);*/

	return 0;
}
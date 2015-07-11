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

#include <gcvl/cuda/cudacore.h>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cout << "Usage: test_cuda path/to/image_left path/to/image_right" << std::endl;
        return 0;
    }

	int dim = 5, maxDisp = 16;
	bool norm = true;
	std::unique_ptr<unsigned char[]> output;

	gcvl::cuda::Core core;

  /*cv::Mat out(bmCPU.getHeight(), bmCPU.getWidth(), CV_8UC1, output.get());

  //cv::namedWindow( "Source Image", cv::WINDOW_AUTOSIZE );// Create a window for display.
  //cv::imshow( "Source Image", image );
  cv::namedWindow( "Disparity Map", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Disparity Map", out );
  cv::waitKey(0);*/

	return 0;

}

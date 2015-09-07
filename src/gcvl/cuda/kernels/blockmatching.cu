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

#include "blockmatching.cuh"
#include "../cudautils.h"
#include "../../gcvlutils.h"

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_BLK 8

//! Calculates a disparity map from two stereoscopic images.
/*!
  \param inputLeft left input image data.
  \param inputRight right input image data.
  \param output output disparity map.
  \param width width of the images.
  \param height height of the images.
  \param dim aggregation window dimension.
  \param radius radius of the aggregation window.
  \param maxdisp maximum disparity.
*/
__global__ void calculateDisparity(const unsigned char * inputLeft, const unsigned char * inputRight,
								   unsigned char * output, const int width, const int height, const int dim,
								   const int radius, const int maxDisp)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int offsetx = x - radius;
	const int offsety = y - radius;

	if (offsetx >= 0 && offsetx + dim < width && offsety >= 0 && offsety + dim < height) {

		unsigned int sum = 0;
		unsigned int bestSum = -1;
		unsigned int bestd = 0;

		for (int d = 0; d < maxDisp; ++d) {
			for (int i = offsety; i < dim + offsety; ++i) {
				for (int j = offsetx; j < dim + offsetx; ++j) {
					if (j - d >= 0)
						sum += abs((int)inputLeft[i * width + j] - (int)inputRight[i * width + j - d]);
					else
						sum += abs((int)inputLeft[i * width + j]);
				}
			}
			if (sum < bestSum) {
				bestSum = sum;
				bestd = d;
			}
			sum = 0;
		}

		output[y * width + x] = bestd;

	}

}

//! Normalizes a disparity map.
/*!
  \param input input disparity map data.
  \param output output disparity map.
  \param width width of the images.
  \param maxdisp maximum disparity.
*/
__global__ void normalizeMap(unsigned char * input, unsigned char *output, const unsigned int width, const int maxDisp) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	output[y * width + x] = (input[y * width + x] / (float)maxDisp) * 255;

}

//! Launches the disparity map computation kernels for two stereoscopic images.
/*!
  \param inputLeft left input image data.
  \param inputRight right input image data.
  \param output output disparity map.
  \param width width of the images.
  \param height height of the images.
  \param dim aggregation window dimension.
  \param radius radius of the aggregation window.
  \param maxdisp maximum disparity.
*/
void launchBM(const unsigned char * inputLeft, const unsigned char * inputRight,
			unsigned char * output, const int width, const int height, const int dim,
			const int radius, const int maxDisp)
{

	dim3 dimBlock(CUDA_BLK, CUDA_BLK);
	dim3 gridSize(roundUpDiv(width,dimBlock.x), roundUpDiv(height,dimBlock.y));
	calculateDisparity<<<gridSize, dimBlock>>>(inputLeft, inputRight, output, width, height, dim, radius, maxDisp);

}

//! Launches the normalization kernels.
/*!
  \param input input disparity map data.
  \param output output disparity map.
  \param width width of the images.
  \param height height of the images.
  \param maxdisp maximum disparity.
*/
void launchNormalization(unsigned char * input, unsigned char * output, const int width, const int height, const int maxDisp)
{

	dim3 dimBlock(CUDA_BLK, CUDA_BLK);
	dim3 gridSize(roundUpDiv(width,dimBlock.x), roundUpDiv(height,dimBlock.y));
	normalizeMap<<<gridSize, dimBlock>>>(input, output, width, maxDisp);

}

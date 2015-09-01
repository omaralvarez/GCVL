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

#pragma once

#include "../export.h"
#include "../gcvlconfig.h"
#include "oclalgorithm.h"

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace gcvl { namespace opencl {

    class GCVL_EXPORT BlockMatching : public Algorithm {

    public:
        BlockMatching(Core & core, std::string inputLeft, std::string inputRight, std::unique_ptr<unsigned char[]> &output);
        ~BlockMatching();
        //! Sets the aggregation window dimension.
        /*!
          \param val aggregation window dimension in pixels.
        */
        inline void setAggDim(const int val) { val % 2 == 0 ? _dim = val + 1 : _dim = val; _radius = _dim / 2; _clDim.Initialize(_dim); _clRadius.Initialize(_radius); }
        //! Sets the maximum disparity.
        /*!
          \param val maximum disparity in pixels.
        */
        inline void setMaxDisp(const int val) { val > 255 ? _maxDisp = 255 : _maxDisp = val; _clMaxDisp.Initialize(_maxDisp); }
        //! Activates or deactivates disparity map normalization.
        /*!
          \param val normalization usage.
        */
        inline void setNormalize(const bool val) { _normalize = val; }
        //! Obtains image width.
        /*!
          \return image width in pixels.
        */
        inline unsigned int getWidth() { return _width; }
        //! Obtains image height.
        /*!
          \return image height in pixels.
        */
        inline unsigned int getHeight() { return _height; }
		void prepare();
		void setArgs();
		void launch();
		void postpare();

    private:
        //Aggregation dimension (odd value)
        int _dim;
        OpenCL_Data<unsigned int> _clDim;
        //Aggregation dimension / 2
        int _radius;
        OpenCL_Data<unsigned int> _clRadius;
        //Maximum disparity
        int _maxDisp;
        bool _normalize;
        OpenCL_Data<unsigned int> _clMaxDisp;
        unsigned int _width;
        OpenCL_Data<unsigned int> _clWidth;
        unsigned int _height;
		OpenCL_Data<unsigned int> _clHeight;
        cv::Mat _inputLeft;
        OpenCL_Array<unsigned char> _clInputLeft;
        cv::Mat _inputRight;
        OpenCL_Array<unsigned char> _clInputRight;
        unsigned char * _output;
        OpenCL_Array<unsigned char> _clOutput;
        OpenCL_Kernel _normalization;

    };

} }

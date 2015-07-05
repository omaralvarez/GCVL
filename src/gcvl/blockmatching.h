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

#include "export.h"
#include "gcvlconfig.h"
#include "algorithm.h"

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace gcvl {

    class GCVL_EXPORT BlockMatching : public Algorithm {

    public:
        BlockMatching(std::string inputLeft, std::string inputRight, std::unique_ptr<unsigned char[]> &output);
        ~BlockMatching();
        inline void setAggDim(const int val) { val % 2 == 0 ? _dim = val + 1 : _dim = val; _radius = _dim / 2; }
        inline void setMaxDisp(const int val) { val > 255 ? _maxDisp = 255 : _maxDisp = val; }
        inline void setNormalize(const bool val) { _normalize = val; }
        inline unsigned int getWidth() { return _width; }
        inline unsigned int getHeight() { return _height; }
		void prepare();
		void setArgs();
		void launch();
		void postpare();
        
    private:
        //Aggregation dimension (odd value)
        int _dim;
        //Aggregation dimension / 2
        int _radius;
        //Maximum disparity
        int _maxDisp;
        bool _normalize;
        unsigned int _width;
        unsigned int _height;
        cv::Mat _inputLeft;
        cv::Mat _inputRight;
        unsigned char * _output;
        
    };

}

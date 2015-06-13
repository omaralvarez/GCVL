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

#include <string>
#include <vector>

namespace gcvl { namespace opencl {

    class GCVL_EXPORT Algorithm {

    public:
        Algorithm();
        virtual ~Algorithm();
		virtual void prepare() = 0;
		virtual void setArgs() = 0;
		virtual void launch() = 0;
		virtual void postpare() = 0;

    protected:
        std::vector<int> kernel;
		//Etc.

    };

} }

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
#include "oclutils.h"

namespace gcvl { namespace opencl {

    class GCVL_EXPORT Core {

    public:
        Core();
		Core(std::string p, bool locking);
        ~Core();
		inline cl_context getContext() { return context; }
		inline cl_device_id getDevice() { return device; }
		inline cl_command_queue getQueue() { return queue; } 

    private:
        OpenCL_platforms_list platforms;
		std::string platform;
		cl_context context;
		cl_device_id device;
		cl_command_queue queue;
		//Etc.

    };

} }

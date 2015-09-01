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
#include "oclutils.h"

namespace gcvl { namespace opencl {

    class GCVL_EXPORT Core {

    public:
        Core();
		Core(std::string platform, bool locking);
    ~Core();
    //! Obtains the associated platform.
		/*!
      \return platform string.
    */
    inline const std::string& getPlatform() { return _platform; }
    //! Obtains the associated context.
		/*!
      \return OpenCL context.
    */
		inline const cl_context& getContext() { return _context; }
    //! Obtains the associated device id.
    /*!
      \return OpenCL device id.
    */
    inline const cl_device_id& getDevice() { return _device; }
    //! Obtains the associated command queue.
    /*!
      \return OpenCL command queue.
    */
		inline const cl_command_queue& getQueue() { return _queue; }
    //! Function that waits for the command queue to finish all tasks.
    inline const void waitForQueue() { clFinish(_queue); }
    //! Function that prints information about the available platforms.
		inline const void printInfo() { _platforms.Print(); }

    private:
        OpenCL_platforms_list _platforms;
		std::string _platform;
		cl_context _context;
		cl_device_id _device;
		cl_command_queue _queue;

    };

} }

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
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 *
 */

#include <GL/glew.h>
//#include <GL/glx.h>
#include <GL/freeglut.h>
#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp> //C++ Bindings .hpp - C Bindings .h
#endif
#ifdef __linux__
#include <GL/glx.h>
#endif

#include <iostream>

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <stdlib.h>
#include <unistd.h>
#endif

#include <pcm/pctree.h>
#include <pcm/l1cache.h>
#include <pcm/pointcloud.h>

#include <dirent.h>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>

#include <omp.h>

typedef enum {
	STATIC = 0,
	DYNAMIC_SEQ = 1,
	DYNAMIC_RAND = 2
} TestMode;

#define L2_SIZE_MB 2000
#define L1_SIZE_MB 1400
#define L1_VBO_SIZE_KB 4096
#define LRU true
#define TEST_MODE DYNAMIC_SEQ
#define TIME 180

//#define CL_PROFILING false

using namespace pcm;

cl::Context context;

const char* oclErrorString(cl_int error) {

    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "";

}

cl::Kernel loadKernel(std::string filename, const char* kernelname, std::vector<cl::Device> & devices) {

	// Read source file
	std::ifstream sourceFile(filename);
		
    std::string sourceCode(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

	//std::cout << "Source: " << std::endl;
	//std::cout << sourceCode << std::endl;

	//Compile kernel
    cl::Program program_ = cl::Program(context, source);
    program_.build(devices,"-cl-nv-verbose");
	
	cl_int err = CL_SUCCESS;
	try {
		return cl::Kernel(program_, kernelname, &err);
	} catch (cl::Error err) {
		std::cout << oclErrorString(err.err()) << std::endl;
        std::cerr 
        << "ERROR: "
        << err.what()
        << "("
        << err.err()
        << ")"
        << std::endl;
	}

	return cl::Kernel();

}

int CLInit(int whichPlatform, std::vector<cl::Device> &selectedDevices) { // -1 means first available platform

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    //if (platforms.size() == 0) return -1;

    std::cout << "*Platforms detected: " << platforms.size() << std::endl;

    int thePlatform = -1;
    std::vector<cl::Device> devices;

    int i = 0;
    for (auto platform : platforms) {

        std::cout << "\n**Platform #" << i << std::endl;
        std::cout << "CL_PLATFORM_PROFILE: " << platform.getInfo<CL_PLATFORM_PROFILE>() << std::endl;
        std::cout << "CL_PLATFORM_VERSION: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
        std::cout << "CL_PLATFORM_NAME: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "CL_PLATFORM_VENDOR: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "CL_PLATFORM_EXTENSIONS: " << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << std::endl;

        cl_int err = CL_SUCCESS;
        try {
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        } catch (cl::Error err) {
            std::cout << "***" << oclErrorString(err.err()) << std::endl;
            std::cerr << "***" << err.what() << "(" << err.err() << ")" << std::endl;

            ++i;
            continue;
        }

        std::cout << "***Devices in platform #" << i << ": " << devices.size() << "\n";

        unsigned int j = 0;
        for (auto device : devices) {

            std::cout << "****Device #" << j << std::endl;
            unsigned bitmap = device.getInfo<CL_DEVICE_TYPE>();
            std::cout << "\t" << "CL_DEVICE_TYPE: ";
            if (bitmap & CL_DEVICE_TYPE_CPU)
                std::cout << "CL_DEVICE_TYPE_CPU";
            if (bitmap & CL_DEVICE_TYPE_GPU) {
                std::cout << "CL_DEVICE_TYPE_GPU";

                if (thePlatform == -1 && (whichPlatform == -1 || whichPlatform == i)) {
                    thePlatform = i;
                    selectedDevices = devices;
                }
            }
            if (bitmap & CL_DEVICE_TYPE_ACCELERATOR)
                std::cout << "CL_DEVICE_TYPE_ACCELERATOR";
            if (bitmap & CL_DEVICE_TYPE_DEFAULT)
                std::cout << "CL_DEVICE_TYPE_DEFAULT";
            // if (bitmap & CL_DEVICE_TYPE_CUSTOM)
            //     std::cout << "CL_DEVICE_TYPE_CUSTOM";
            std::cout << "\n";

            std::cout << "\t" << "CL_DEVICE_VENDOR_ID: " << device.getInfo<CL_DEVICE_VENDOR_ID>() << std::endl;
            std::cout << "\t" << "CL_DEVICE_MAX_COMPUTE_UNITS: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\t" << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
            std::cout << "\t" << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

            ++j;
        }

        ++i;
    }

    std::cout << std::endl << "\n->Selecting GPU *Device #0* on Platform #" << thePlatform << std::endl << std::endl;

    try {
	//Init context depending on OS
	#if defined (__APPLE__) || defined(MACOSX)

		CGLContextObj kCGLContext = CGLGetCurrentContext();
		CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
		cl_context_properties properties[] =
			{
				CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
				0
			};
		//Apple's implementation is weird, and I don't the default values assumed by cl.hpp will work
		//this works
		//cl_context cxGPUContext = clCreateContext(props, 0, 0, NULL, NULL, &err);
		//these dont
		//cl_context cxGPUContext = clCreateContext(props, 1,(cl_device_id*)&devices.front(), NULL, NULL, &err);
		//cl_context cxGPUContext = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
        context = cl::Context(properties);   //We may have to edit line 1448 of cl.hpp to add this constructor

	#else
		#if defined WIN32 // Win32

			cl_context_properties properties[] =
				{
					CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
					CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
                                            CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[thePlatform])(),
					0
				};
			context =  cl::Context(CL_DEVICE_TYPE_ALL, properties);

		#else

                        context = cl::Context(selectedDevices, NULL, NULL, NULL, NULL);

		#endif
	#endif
    } catch (cl::Error err) {
        std::cout << oclErrorString(err.err()) << std::endl;
        std::cerr << err.what() << "(" << err.err() << ")" << std::endl;
        return -1;
    }

    return thePlatform;
}

template<class PointCloud>
double getMaxLevelCPU(PointCloud & cloud) {

	osg::Timer_t tIni = osg::Timer::instance()->tick();

	cloud.getMaxLevel();

	return osg::Timer::instance()->delta_s(tIni, osg::Timer::instance()->tick()); 

}

template<class PointCloud>
double getMaxLevelGPU(PointCloud & cloud, std::vector<cl::Device> & devices) {

	osg::Timer_t tIni = osg::Timer::instance()->tick();

	cl_int err = CL_SUCCESS;
    try {
                cl::Kernel kernel = loadKernel("parallel_reduction.cl","parallel_reduction", devices);
		cl::CommandQueue queue(context, devices[0], 0, &err);
		/*void * foo = malloc(sizeof(float) * 3 * 3 + sizeof(unsigned char) * 3 * 4); 

		float pos[3*3] = { 1,1,1,2,2,2,3,3,3 }; 
		unsigned char col[3*4] = { 1,1,1,1,2,2,2,1,3,3,3,1 }; 

		float * dest = (float*)(foo); 

		memcpy(foo,pos,3*3*sizeof(float));
		memcpy(dest+3*3,col,sizeof(unsigned char)*3*4);

		unsigned char * col2 = (unsigned char *)(dest+3*3);
		//uchar result = ((uchar *)(dest+3*3))[3];
		unsigned char result = col2[3];
		std::cout << (unsigned int)result << (unsigned int)col2[4] << std::endl;*/
		//unsigned int total_points = 0;
		float result = -FLT_MAX; 
		size_t localWorkSize = 64; //OpenCL worksize 

		PCTree::NodeIDsList list;

		unsigned int index = 0;
		while (cloud.getNextDataGPU(0.f, NULL, list, index, 0, 0) != 0) {
			for (PCTree::NodeIDsList::const_iterator i = list.begin(); i != list.end(); ++i) {
				//std::cout << *i << std::endl;
				//++step;
				const auto * VBOdata = cloud.getL1Entry(*i); //Get VRAM cache info to draw

				if (VBOdata) {

					const std::vector<GLuint> VBOs = std::get<0>(*VBOdata);
					const unsigned int VBOSize = std::get<1>(*VBOdata);
					const unsigned int lastVBOSize = std::get<2>(*VBOdata);

					for (unsigned int j = 0; j < VBOs.size()-1; ++j) {
						//std::cout << "Points in VBO: " << VBOSize << std::endl;
						//total_points += VBOSize;
						int numWorkGroups = (VBOSize + localWorkSize - 1) / localWorkSize; // round up
						size_t globalWorkSize = numWorkGroups * localWorkSize; // must be evenly divisible by localWorkSize
						//clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

						//Make VBO accessible by OpenCL
						//glFinish();
						std::vector<cl::Memory> cl_vbos;
						cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, VBOs[j], &err)); //Read only would improve perf

						cl::Buffer odata = cl::Buffer(context, CL_MEM_WRITE_ONLY, VBOSize * sizeof(float));
						//queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), A);

						//Pass the VBO argument to the kernel
						kernel.setArg(0, cl_vbos[0]);
						kernel.setArg(1, odata);
						kernel.setArg(2, VBOSize);
						kernel.setArg(3, cl::__local(sizeof(float)*localWorkSize));

						//glFlush();
						glFinish();

						//Run the kernel
						cl::Event event;

						queue.enqueueAcquireGLObjects(&cl_vbos, NULL, &event);
						//queue.finish();

						queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), NULL, &event);
						//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange, NULL, &event);
						//queue.finish();

						queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);

						float * odatar = new float[numWorkGroups];

						queue.enqueueReadBuffer(odata, CL_TRUE, 0, numWorkGroups * sizeof(float), odatar);
						//std::cout << numWorkGroups << std::endl;
						//float max2 = -FLT_MAX;
						for (int k = 0; k < numWorkGroups; ++k) {
							/*if(max2 < odatar[k])
								max2 = odatar[k];*/
							if(result < odatar[k]) 
								result = odatar[k];
						}
						delete [] odatar;
						//queue.finish();

						/*float * vertices2; //= new float[lastVBOSize];
						glBindBuffer( GL_ARRAY_BUFFER, VBOs[j] );
						//glGetBuffer
						vertices2 = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
						//glGetBufferSubData ( GL_ARRAY_BUFFER, 0, lastVBOSize * sizeof( float ) * 3, vertices2 );
						//glBindBuffer( GL_ARRAY_BUFFER, 0 );
						float max = -FLT_MAX;
						for ( unsigned int l = 0; l < VBOSize; ++l ) {
							//std::cout << "point " << i << ": " << vertices2[ i * 3 + 0 ] << " / " << vertices2[ i * 3 + 1 ] << " / " << vertices2[ i * 3 + 2 ] << std::endl;
							//max = std::max(max,vertices2[ i * 3 + 1 ]);
							if(max < vertices2[ l * 3 + 1 ])
								max = vertices2[ l * 3 + 1 ];
						}
						glUnmapBuffer(GL_ARRAY_BUFFER);
						if(max != max2) std::cout << "Error al calcular la cota MAX: " << max << " " << max2 << std::endl;*/
						//event.wait(); //I'm not sure this is necessary

					}
					//total_points += lastVBOSize;
					//std::cout << "Points in last VBO: " << lastVBOSize << std::endl;
					int numWorkGroups = (lastVBOSize + localWorkSize - 1) / localWorkSize; // round up
					size_t globalWorkSize = numWorkGroups * localWorkSize; // must be evenly divisible by localWorkSize
					//Make VBO accessible by OpenCL
					//glFinish();
					std::vector<cl::Memory> cl_vbos;
					cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, VBOs.back(), &err));//Read only would improve perf

					cl::Buffer odata = cl::Buffer(context, CL_MEM_WRITE_ONLY, lastVBOSize * sizeof(float)); //Solo se necesitan numworkgroups en vez de 1 por cada punto
					//queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), A);

					//Pass the VBO argument to the kernel
					kernel.setArg(0, cl_vbos[0]);
					kernel.setArg(1, odata);
					kernel.setArg(2, lastVBOSize);
					kernel.setArg(3, cl::__local(sizeof(float)*localWorkSize));

					//glFlush();
					glFinish();

					//Run the kernel
					cl::Event event;
					//cl::CommandQueue queue(context, devices[0], 0, &err);

					queue.enqueueAcquireGLObjects(&cl_vbos, NULL, &event);
					//queue.finish();

					queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), NULL, &event);
					//queue.finish();

					queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);

					float * odatar = new float[numWorkGroups];
					queue.enqueueReadBuffer(odata, CL_TRUE, 0, numWorkGroups * sizeof(float), odatar);
					//float max2 = -FLT_MAX;
					for (int k = 0; k < numWorkGroups; ++k) {
						//max2 = std::max(max2,odatar[k]);
						/*if(max2 < odatar[k])
							max2 = odatar[k];*/
						if(result < odatar[k]) 
							result = odatar[k];
					}
					delete [] odatar;
					//queue.finish();
					//glBindBuffer(GL_ARRAY_BUFFER, VBOs.back());
					/*float * vertices2; //= new float[lastVBOSize];
					glBindBuffer( GL_ARRAY_BUFFER, VBOs.back() );
					//glGetBuffer
					vertices2 = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
					//glGetBufferSubData ( GL_ARRAY_BUFFER, 0, lastVBOSize * sizeof( float ) * 3, vertices2 );
					//glBindBuffer( GL_ARRAY_BUFFER, 0 );
					float max = -FLT_MAX;
					for ( unsigned int l = 0; l < lastVBOSize; ++l ) {
						//std::cout << "point " << i << ": " << vertices2[ i * 3 + 0 ] << " / " << vertices2[ i * 3 + 1 ] << " / " << vertices2[ i * 3 + 2 ] << std::endl;
						//max = std::max(max,vertices2[ i * 3 + 1 ]);
						if(max < vertices2[ l * 3 + 1 ])
							max = vertices2[ l * 3 + 1 ];
					}
					glUnmapBuffer(GL_ARRAY_BUFFER);*
					//glFinish();
					//delete [] vertices2;

					//queue.finish();
					//event.wait(); //I'm not sure this is necessary
					if(max != max2) std::cout << "Error al calcular la cota MAX: " << max << " " << max2 << std::endl;*/
					//std::cout << lastVBOSize << std::endl;

				}
		
			}
		}

		//std::cout << "GPU Result: " << result << std::endl;
		//std::cout << "(Time: " << time << " s.)" << std::endl;
		
		//std::cout << step << std::endl;
		//std::cout << "Total points processed: " << total_points << std::endl;

	} catch (cl::Error err) {
		std::cout << oclErrorString(err.err()) << std::endl;
		std::cerr 
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
	}

	return osg::Timer::instance()->delta_s(tIni, osg::Timer::instance()->tick());

}

template<class PointCloud>
    void CLTest(PointCloud & cloud, std::vector<cl::Device> & devices) {

	cl_int err = CL_SUCCESS;
    try {
                //cl::Kernel kernel = loadKernel("test.cl","test", devices);
                cl::Kernel kernel = loadKernel("change_VBO_color.cl","change_VBO_color", devices);
		cl::CommandQueue queue(context, devices[0], 0, &err);
		/*void * foo = malloc(sizeof(float) * 3 * 3 + sizeof(unsigned char) * 3 * 4); 

		float pos[3*3] = { 1,1,1,2,2,2,3,3,3 }; 
		unsigned char col[3*4] = { 1,1,1,1,2,2,2,1,3,3,3,1 }; 

		float * dest = (float*)(foo); 

		memcpy(foo,pos,3*3*sizeof(float));
		memcpy(dest+3*3,col,sizeof(unsigned char)*3*4);

		unsigned char * col2 = (unsigned char *)(dest+3*3);
		//uchar result = ((uchar *)(dest+3*3))[3];
		unsigned char result = col2[3];
		std::cout << (unsigned int)result << (unsigned int)col2[4] << std::endl;*/
		//unsigned int total_points = 0;
		float result = -FLT_MAX; 
		size_t localWorkSize = 64; //OpenCL worksize 

		PCTree::NodeIDsList list;

		osg::Timer_t tIni = osg::Timer::instance()->tick();
		unsigned int index = 0;
		while (cloud.getNextDataGPU(0.f, NULL, list, index, 0, 0) != 0) {
			for (PCTree::NodeIDsList::const_iterator i = list.begin(); i != list.end(); ++i) {
				//std::cout << *i << std::endl;
				//++step;
				auto * VBOdata = cloud.getL1Entry(*i); //Get VRAM cache info to draw

				if (VBOdata) {

					const std::vector<GLuint> VBOs = std::get<0>(*VBOdata);
					const unsigned int VBOSize = std::get<1>(*VBOdata);
					const unsigned int lastVBOSize = std::get<2>(*VBOdata);
					cloud.setChunkWriteL1(*i);

					for (unsigned int j = 0; j < VBOs.size()-1; ++j) {
						//std::cout << "Points in VBO: " << VBOSize << std::endl;
						//total_points += VBOSize;


						//Make VBO accessible by OpenCL
						//glFinish();
						std::vector<cl::Memory> cl_vbos;
						cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, VBOs[j], &err)); //Read only would improve perf

						//Pass the VBO argument to the kernel
						kernel.setArg(0, cl_vbos[0]);
						cl_uchar4 color = { 255, 255, 0, 255};
						kernel.setArg(1, color);

						//glFlush();
						glFinish();

						//Run the kernel
						cl::Event event;

						queue.enqueueAcquireGLObjects(&cl_vbos, NULL, &event);
						//queue.finish();

						queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VBOSize), cl::NullRange, NULL, &event);
						//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange, NULL, &event);
						//queue.finish();

						queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);
						queue.finish();

						/*float * vertices2; //= new float[lastVBOSize];
						glBindBuffer( GL_ARRAY_BUFFER, VBOs[j] );
						//glGetBuffer
						vertices2 = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
						//glGetBufferSubData ( GL_ARRAY_BUFFER, 0, lastVBOSize * sizeof( float ) * 3, vertices2 );
						//glBindBuffer( GL_ARRAY_BUFFER, 0 );
						float max = -FLT_MAX;
						for ( unsigned int l = 0; l < VBOSize; ++l ) {
							//std::cout << "point " << i << ": " << vertices2[ i * 3 + 0 ] << " / " << vertices2[ i * 3 + 1 ] << " / " << vertices2[ i * 3 + 2 ] << std::endl;
							//max = std::max(max,vertices2[ i * 3 + 1 ]);
							if(max < vertices2[ l * 3 + 1 ])
								max = vertices2[ l * 3 + 1 ];
						}
						glUnmapBuffer(GL_ARRAY_BUFFER);
						if(max != max2) std::cout << "Error al calcular la cota MAX: " << max << " " << max2 << std::endl;*/
						//event.wait(); //I'm not sure this is necessary

					}
					//total_points += lastVBOSize;
					//std::cout << "Points in last VBO: " << lastVBOSize << std::endl;

					//Make VBO accessible by OpenCL
					//glFinish();
					std::vector<cl::Memory> cl_vbos;
					cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, VBOs.back(), &err));//Read only would improve perf

					//Pass the VBO argument to the kernel
					kernel.setArg(0, cl_vbos[0]);
					cl_uchar4 color = { 255, 255, 0, 255};
					kernel.setArg(1, color);

					//glFlush();
					glFinish();

					//Run the kernel
					cl::Event event;
					//cl::CommandQueue queue(context, devices[0], 0, &err);

					queue.enqueueAcquireGLObjects(&cl_vbos, NULL, &event);
					//queue.finish();

					queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(lastVBOSize), cl::NullRange, NULL, &event);
					//queue.finish();

					queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);
					queue.finish();
					//glBindBuffer(GL_ARRAY_BUFFER, VBOs.back());
					/*float * vertices2; //= new float[lastVBOSize];
					glBindBuffer( GL_ARRAY_BUFFER, VBOs.back() );
					//glGetBuffer
					vertices2 = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
					//glGetBufferSubData ( GL_ARRAY_BUFFER, 0, lastVBOSize * sizeof( float ) * 3, vertices2 );
					//glBindBuffer( GL_ARRAY_BUFFER, 0 );
					float max = -FLT_MAX;
					for ( unsigned int l = 0; l < lastVBOSize; ++l ) {
						//std::cout << "point " << i << ": " << vertices2[ i * 3 + 0 ] << " / " << vertices2[ i * 3 + 1 ] << " / " << vertices2[ i * 3 + 2 ] << std::endl;
						//max = std::max(max,vertices2[ i * 3 + 1 ]);
						if(max < vertices2[ l * 3 + 1 ])
							max = vertices2[ l * 3 + 1 ];
					}
					glUnmapBuffer(GL_ARRAY_BUFFER);*
					//glFinish();
					//delete [] vertices2;

					//queue.finish();
					//event.wait(); //I'm not sure this is necessary
					if(max != max2) std::cout << "Error al calcular la cota MAX: " << max << " " << max2 << std::endl;*/
					//std::cout << lastVBOSize << std::endl;

				}
		
			}
		}

		std::cout << "GPU Result: " << result << std::endl;
		std::cout << "(Time: " << osg::Timer::instance()->delta_s(tIni, osg::Timer::instance()->tick()) << " s.)" << std::endl;
		//std::cout << step << std::endl;
		//std::cout << "Total points processed: " << total_points << std::endl;

	} catch (cl::Error err) {
		std::cout << oclErrorString(err.err()) << std::endl;
		std::cerr 
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
	}

}

double gpuExecutionTime(cl::Event &event) {

	cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	return (double)1.0e-9 * (end - start);

}

template<class PointCloud>
float getRadius(const osg::Vec3 & coord, PointCloud & cloud, cl::Kernel & kernel, cl::CommandQueue & queue, cl_int & err) {

	//unsigned int total_points = 0;
	unsigned int numNeighbours = 4;
	float max_dist = 0.01f;
	std::multiset<float> kNearest;
	size_t localWorkSize = 64; //OpenCL worksize 
	//size_t localWorkSize = 8; //OpenCL worksize 

	PCTree::NodeIDsList nList;
	PCTree::NodeIDsList fullList;
	//osg::Timer_t tIni2 = osg::Timer::instance()->tick();
	PCTree::NodeIDsList::const_iterator index;
	while(cloud.getNeighborDataGPU(coord, max_dist, nList, fullList, index) != 0) {
		for (PCTree::NodeIDsList::const_iterator i = nList.begin(); i != nList.end(); ++i) {
			//std::cout << "->" <<  *i << std::endl;
			auto * VBOdata = cloud.getL1Entry(*i); //Get VRAM cache info to draw

			if (VBOdata) {

				const std::vector<GLuint> VBOs = std::get<0>(*VBOdata);
				//std::cout << "VBOs: " << VBOs.size() << std::endl;
				const unsigned int VBOSize = std::get<1>(*VBOdata);
				const unsigned int lastVBOSize = std::get<2>(*VBOdata);
				//cloud.setChunkWriteL1(*i);
				int numWorkGroups = (VBOSize + localWorkSize - 1) / localWorkSize; // round up
				size_t globalWorkSize = numWorkGroups * localWorkSize; // must be evenly divisible by localWorkSize

				cl_float3 c = { coord.x(), coord.y(), coord.z() };
				cl::Buffer odata = cl::Buffer(context, CL_MEM_WRITE_ONLY, VBOSize * sizeof(float));
				float * odatar = new float[VBOSize];

				for (unsigned int j = 0; j < VBOs.size()-1; ++j) {
					//std::cout << "Points in VBO: " << VBOSize << std::endl;
					//total_points += VBOSize;

					std::vector<cl::Memory> cl_vbos;
					glFinish();
					cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_ONLY, VBOs[j], &err));//Read only would improve perf

					kernel.setArg(0, cl_vbos[0]);
					kernel.setArg(1, odata); //TODO only cl_vbos and VBO size change so is the only ting that we have to call with setArg each time
					kernel.setArg(2, VBOSize);
					kernel.setArg(3, cl::__local(sizeof(float)*localWorkSize));
					kernel.setArg(4, c);

					queue.enqueueAcquireGLObjects(&cl_vbos, NULL, NULL);
					//queue.finish();

					queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), NULL, NULL);

					queue.enqueueReleaseGLObjects(&cl_vbos, NULL, NULL);

					//glFinish();
					//queue.finish();

					queue.enqueueReadBuffer(odata, CL_TRUE, 0, VBOSize * sizeof(float), odatar);
					//std::cout << numWorkGroups << std::endl;
					//float max2 = -FLT_MAX;
					/*std::cout << VBOSize << std::endl;
					std::cout << "Distances: ";
					for (unsigned int k = 0; k < VBOSize; k++){
						std::cout << odatar[k] << " ";
						if(k == 7) std::cout << "|" ;
					}
					std::cout << std::endl;*/
					for (int k = 0; k < numWorkGroups; ++k) {
						for (unsigned int l = 0; l < numNeighbours && k*localWorkSize + l < VBOSize; ++l) {
							//std::cout << "Distance: " << odatar[k] << std::endl;
							float val = odatar[k*localWorkSize+l];
							if (kNearest.size() < numNeighbours) kNearest.insert(val);
							else {
								std::multiset<float>::iterator it = kNearest.end();
								--it;
								if (*it > val) {
									kNearest.erase(it);
									kNearest.insert(kNearest.end(),val);
								}
							}
						//std::cout << odatar[k] << " ";
						//if(k % 64 == 0) std::cout << " | ";
						}
					}
					//delete [] odatar;
					//std::cout << std::endl;

					//glFinish();
					//event.wait(); //I'm not sure this is necessary

				}
				//total_points += lastVBOSize;
				//std::cout << "Distance: " << *kNearest.rbegin() << std::endl;

				//Make VBO accessible by OpenCL
				//glFinish();
				numWorkGroups = (lastVBOSize + localWorkSize - 1) / localWorkSize; // round up
				globalWorkSize = numWorkGroups * localWorkSize; // must be evenly divisible by localWorkSize

				std::vector<cl::Memory> cl_vbos;
				glFinish();
				cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_ONLY, VBOs.back(), &err));//Read only would improve perf
				//queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), A);

				//Pass the VBO argument to the kernel

				kernel.setArg(0, cl_vbos[0]);
				kernel.setArg(1, odata);
				kernel.setArg(2, lastVBOSize);
				kernel.setArg(3, cl::__local(sizeof(float)*localWorkSize));
				kernel.setArg(4, c);

				queue.enqueueAcquireGLObjects(&cl_vbos, NULL, NULL);
				//glFlush();
				//queue.finish();
				//queue.finish();
				//cl::Event event;
				//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), NULL, &event);
				queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), NULL, NULL);
				//event.wait();

				//std::cout << "Kernel ex. time: " << gpuExecutionTime(event) << std::endl;

				//delete event;
				
				//queue.finish();

				queue.enqueueReleaseGLObjects(&cl_vbos, NULL, NULL);

				//glFinish();
				//queue.finish();
				//osg::Timer_t tIni2 = osg::Timer::instance()->tick();
				//float * odatar = new float[lastVBOSize];

				queue.enqueueReadBuffer(odata, CL_TRUE, 0, lastVBOSize * sizeof(float), odatar);
				//queue.finish();
			
				//std::cout << lastVBOSize << std::endl;
				//float max2 = -FLT_MAX;
				/*std::cout << "Distances: ";
				for (unsigned int k = 0; k < lastVBOSize; k++){
					std::cout << odatar[k] << " ";
					if(k == 7) std::cout << "|" ;
				}
				std::cout << std::endl;*/

				for (int k = 0; k < numWorkGroups; ++k) {
					for (unsigned int l = 0; l < numNeighbours && k*localWorkSize + l < lastVBOSize; ++l) {
						//std::cout << "Distance: " << odatar[k] << std::endl;
						float val = odatar[k*localWorkSize+l];
						if (kNearest.size() < numNeighbours) kNearest.insert(val);
						else {
							std::multiset<float>::iterator it = kNearest.end();
							--it;
							if (*it > val) {
								kNearest.erase(it);
								kNearest.insert(kNearest.end(),val);
							}
						}
					//std::cout << odatar[k] << " ";
					}	
				}
				//
				//std::cout << std::endl << " *** " << " *** (Time: " << osg::Timer::instance()->delta_s(tIni2, osg::Timer::instance()->tick()) << " s.)" << std::endl;
				//std::cout << std::endl;
				delete [] odatar;	
			}
								
		}

	}
	
	/*for (auto k : kNearest) {
		std::cout << k << std::endl;
	}*/
	//std::cout << std::endl << " *** " << " *** (Time: " << osg::Timer::instance()->delta_s(tIni2, osg::Timer::instance()->tick()) << " s.)" << std::endl;
	return *kNearest.rbegin();

}

template<class PointCloud>
float getRadiusCPU(const osg::Vec3 & coord, PointCloud & cloud) {

	//unsigned int total_points = 0;
	unsigned int numNeighbours = 4;
	float max_dist = 0.01f;
	std::multiset<float> kNearest;
	//char * tmpChunkBuffer = (char *)malloc(cloud.getMaxChunkSize());

	PCTree::NodeIDsList list;
	cloud.getNeighborData(coord, max_dist, list);
	//std::cout << std::endl << " *** " << " *** (Time: " << osg::Timer::instance()->delta_s(tIni2, osg::Timer::instance()->tick()) << " s.)" << std::endl;
	for (PCTree::NodeIDsList::const_iterator i = list.begin(); i != list.end(); ++i) {

		Chunk c;
		//c.data = tmpChunkBuffer;
		cloud.getChunkL2(*i, c);
		//cloud.setChunkUsedL2(*i);
		//cloud.chunkInfo(*i); 
		//cloud.getChunkL2(*i, c);
		/*std::cout << chunkID << " -> " << c.nPoints << std::endl;
		osg::BoundingBox bb = tree.getNode(chunkID).bb;
		std::cout << bb.center().x() << " " << bb.center().y() << " " << bb.center().z() << " " << bb.radius()  << std::endl;*/
		//osg::Timer_t tIni = osg::Timer::instance()->tick();
			
		for (unsigned int k = 0; k < c.nPoints; k++) {
			osg::Vec3 neighbor = c.getCoords(k);
			if (coord == neighbor) continue;
			//std::cout << neighbor.x() << " " << neighbor.y() << " " << neighbor.z() << std::endl;
			float neighInfo = (neighbor - coord).length();

			if (kNearest.size() < numNeighbours){
				kNearest.insert(neighInfo);
			} else {
				std::multiset<float>::iterator it = kNearest.end();
				--it;
				if (*it > neighInfo) {
					kNearest.erase(it);
					kNearest.insert(kNearest.end(),neighInfo);
				}
			}
			//if (i==0){std::cout << neighInfo << std::endl;for (std::multiset<float>::iterator h = kNearest.begin(); h != kNearest.end(); h++) std::cout << "*" <<*h <<"*"<< std::endl;}
		}

		cloud.freeChunkL2(*i);

	}
	
	//free(tmpChunkBuffer);

	/*for (auto k : kNearest) {
		std::cout << k << std::endl;
	}*/
	//std::cout << std::endl << " *** " << " *** (Time: " << osg::Timer::instance()->delta_s(tIni2, osg::Timer::instance()->tick()) << " s.)" << std::endl;
	return *kNearest.rbegin();

}

template<class PointCloud>
double radiiEstimationCPU(PointCloud & cloud) {

	PCTree::NodeIDsList list;

	osg::Timer_t tIni = osg::Timer::instance()->tick();
	cloud.getData(0.f, NULL, list, 0, 0);
	for (PCTree::NodeIDsList::const_iterator i = list.begin(); i != list.end(); ++i) {

		std::cout << *i << std::endl;

		Chunk c;
		//Get chunk
		cloud.getChunkL2(*i, c);
		//Lock the chunk so its not evicted
		//cloud.setChunkUsedL2(*i);
		//cloud.chunkInfo(*i); 

		#pragma omp parallel for
		for (int64_t j = 0; j < (int64_t)c.nPoints; j++) {
			//std::cout << "Thread num: " << omp_get_thread_num() << std::endl;
			//if(j==0 || j==1){
			/*std::cout << omp_get_thread_num() << " Before lock!" << std::endl;
			cloud.mutexLock();
			std::cout << omp_get_thread_num() << " After lock!" << std::endl;*/
			//}
			osg::Vec3 coord = c.getCoords(j);
			float radius = getRadiusCPU(coord,cloud);
			float * data = (float *)((char *)c.getData(j)+3);
			*data = radius;
					
		}
			
		//No longer in use
		cloud.freeChunkL2(*i);
		//Set chunk as written
		cloud.setChunkWriteL2(*i);
								
	}


	return osg::Timer::instance()->delta_s(tIni, osg::Timer::instance()->tick());

}

template<class PointCloud>
double radiiEstimationGPU(PointCloud & cloud, std::vector<cl::Device> & devices) {

	cl_int err = CL_SUCCESS;
    try {
                cl::Kernel kernel = loadKernel("bitonic_sort.cl","bitonic_sort", devices);
#ifdef CL_PROFILING 
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err); 
#else
		cl::CommandQueue queue(context, devices[0], 0, &err); 
#endif
		//cl::Event event;


		PCTree::NodeIDsList list;

		osg::Timer_t tIni = osg::Timer::instance()->tick();
		unsigned int index = 0;
		while (cloud.getNextDataGPU(0.f, NULL, list, index, 0, 0) != 0) {
			for (PCTree::NodeIDsList::const_iterator i = list.begin(); i != list.end(); ++i) {

				std::cout << *i << std::endl;

				Chunk c;
				cloud.getChunkL2(*i, c);

				for (unsigned int j = 0; j < c.nPoints; j++) {

					osg::Vec3 coord = c.getCoords(j);
					//osg::Timer_t tIni2 = osg::Timer::instance()->tick();
					//std::cout << neighbor.y() << std::endl;
					//max = std::max(max,neighbor.y());
					//std::cout << "Point: " << j << std::endl;
					float radius = getRadius(coord,cloud,kernel,queue,err);
					float * data = (float *)((char *)c.getData(j)+3);
					*data = radius;
					//std::cout << "Radius: " << radius << std::endl;
					//std::cout << std::endl << " *** " << j << " *** (Time: " << osg::Timer::instance()->delta_s(tIni2, osg::Timer::instance()->tick()) << " s.)" << std::endl;
					//std::cout << "After asig: " << (unsigned int)data[0] << " " << (unsigned int)data[1] << " " << (unsigned int)data[2] << std::endl;
					
				}
				
				cloud.setChunkWriteL2(*i);
				cloud.freeChunkL2(*i);
								
			}
		}

		/*BinPCHandler * bpch = new BinPCHandler(BinPCHandler::OUTPUT, "C:/Users/Omar/Documents/Datasets/CLRadius.bpc", 7, true, false);

		cloud.save(bpch);

		bpch->close();
		delete bpch;*/

		//std::cout << "GPU Result: " << result << std::endl;
		return osg::Timer::instance()->delta_s(tIni, osg::Timer::instance()->tick());
		//std::cout << step << std::endl;
		//std::cout << "Total points processed: " << total_points << std::endl;

	} catch (cl::Error err) {
		std::cout << oclErrorString(err.err()) << std::endl;
		std::cerr 
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
	}

	return -1.;

}

void help() {

	std::cout << std::endl << "Usage: test_cl <platform> <dataset> [<l2size> <l1size> <l1vbosize> <hashmap> <lru>] <output>" << std::endl;
	std::cout << std::endl;
	std::cout << "	<platform>:	        Platform to use (integer). Run without params to see available platforms." << std::endl;
	std::cout << "	<dataset>:		Path to the dataset folder." << std::endl;
	std::cout << "	<l2size>:		Size of L2 cache in MB." << std::endl;
	std::cout << "	<l1size>:		Size of L1 cache in MB." << std::endl;
	std::cout << "	<l1vbosize>:		Size of VBOs in KB." << std::endl;
	std::cout << "	<hashmap>:		Use hashmap: true or false." << std::endl;
	std::cout << "	<lru>:			Use lru: true or false." << std::endl;
	std::cout << "	<output>:		Path to the output folder." << std::endl;
	std::cout << std::endl;
	std::cout << "Example:	test_cl /path/to/dataset /path/to/output" << std::endl;
	std::cout << "		test_cl /path/to/dataset 2000 600 1024 true false /path/to/output" << std::endl;
	std::cout << std::endl;

}

void output(std::string output, CacheStatsManager::CacheStats s1, unsigned int l2size, unsigned int l1size, unsigned int l1vbosize, std::string hashmap, bool lru, double time, double timeCPU){

	std::cout << "Radii estimation time GPU: " << time << " s." << std::endl;
	std::cout << "Radii estimation time CPU: " << timeCPU << " s." << std::endl;
	std::cout << "TD aleat: " << 100.0*s1.hits/(float)s1.requests << " % hits" << std::endl;
	std::cout << "    Avg Loading Time: " << s1.averageLoadingTime << " ms." << std::endl;
	std::cout << "    Avg Completness Level: " << 100*s1.averageCompletnessLevel << " %" << std::endl;
	std::cout << "    Avg Max Level: " << s1.averageMaxLevel << std::endl;
	std::cout << "    Avg Miss Penalty: " << (s1.misses/(float)s1.requests)*s1.averageLoadingTime << " ms." << std::endl;
	std::cout << "    Capacity Misses: " << 100*s1.capacity/(float)s1.requests << " %" << std::endl;
	std::cout << "    Compulsory Misses: " << 100*s1.compulsory/(float)s1.requests << " %" << std::endl;
	std::cout << std::endl;

	std::string dirName(output+std::string("/"));

	DIR *direc;
	struct dirent *ent;
	unsigned int count = 0;
	if ((direc = opendir (dirName.c_str())) != NULL) {
	  /* print all the files and directories within directory */
		while ((ent = readdir (direc)) != NULL) {
			//printf ("%s\n", ent->d_name);
			count++;
		}
	  closedir (direc);
	} else {
	  /* could not open directory */
	  perror ("Could not open dir.");
	}

	std::ostringstream oss;
	oss << "benchmark_" << std::setfill('0') << std::setw(4) << count-1 << "_" << l2size << "_" << l1size << "_" << l1vbosize << "_" << hashmap << "_";
	if(lru) oss << "LRU";
	else oss << "NOLRU";
	oss << ".dat";

	std::string fileName = oss.str();
	std::cout << "Saving benchmark output to " << fileName << std::endl;
	std::ofstream myfile;
	myfile.open (dirName + fileName);
	if (myfile.is_open()) {
		myfile << "#Time GPU: s" << std::endl;
		myfile << "#Time CPU: s" << std::endl;
		myfile << "#Cache hits: %" << std::endl;
		myfile << "#Avg Loading Time: ms" << std::endl;
		myfile << "#Avg Completness Level: %" << std::endl;
		myfile << "#Avg Max Level: tree level" << std::endl;
		myfile << time << std::endl;
		myfile << timeCPU << std::endl;
		myfile << 100.0*s1.hits/(float)s1.requests << std::endl;
		myfile << s1.averageLoadingTime << std::endl;
		myfile << 100.0*s1.averageCompletnessLevel << std::endl;
		myfile << s1.averageMaxLevel << std::endl;
		myfile << (s1.misses/(float)s1.requests)*s1.averageLoadingTime << std::endl;
		myfile << 100*s1.capacity/(float)s1.requests << std::endl;
		myfile << 100*s1.compulsory/(float)s1.requests << std::endl;
		myfile.close();//TODO store radii estimation time
	} else std::cout << "Unable to open file" << std::endl;

}

bool getBool(std::string lruBool) {

	if(lruBool == std::string("true")) return true;
	else if(lruBool == std::string("false")) return false;
		else return false;

}

int main(int argc, char** argv)
{

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(300,300);
    glutCreateWindow(argv[0]);
	glutHideWindow();
	srand((unsigned int) pow((float)(time(NULL)%100),3.0f));
	/*GLenum err = glewInit();
	if (GLEW_OK != err) {
		//If init fails, there is probably no GL context, or other error.
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }*/

        std::vector<cl::Device> devices;
	
	std::cout << "Using " << omp_get_max_threads() << " threads in computations!" << std::endl;

	omp_set_num_threads(omp_get_max_threads());

	try {

		//Load point cloud
		if(argc==1) {
                    if (CLInit(-1, devices) == -1) {
                        std::cout << "Uncapable of finding a valid GPU Device!" << std::endl;
                        return 1;
                    }
                    help();
                    return 1;
		} else {
                    if (CLInit(atoi(argv[1]), devices) == -1) {
                        std::cout << "Uncapable of using platform #" << atoi(argv[1]) << std::endl;
                        return 1;
                    }
                
                    if (argc == 4) {
			PointCloud<std::unordered_map> * cloud = new PointCloud<std::unordered_map>(std::string(""),std::string(argv[2])+"/",L2_SIZE_MB,L1_SIZE_MB,L1_VBO_SIZE_KB);

			//double timeCPU = radiiEstimationCPU(*cloud);
			double timeCPU = getMaxLevelCPU(*cloud);
			//double timeCPU = cloud.radiiEstimation();
			//double timeCPU = cloud.voxelGridFilter(0.001f,0.001f,0.001f,"lour_VOX_t1.bpc");
			//double timeCPU = cloud.statisticalOutlierRemoval(50,1.f,"C:/Users/Omar/Documents/Datasets/test_stat2.bpc");
			//double timeGPU = radiiEstimationGPU(*cloud, devices);
			double timeGPU = getMaxLevelGPU(*cloud, devices);
			//double timeGPU = 0.;
			std::cout << "(Time CPU/GPU: " << timeCPU << "/" << timeGPU << " s.)" << std::endl;
			output(argv[3],cloud->getL1Stats(),L2_SIZE_MB,L1_SIZE_MB,L1_VBO_SIZE_KB,std::string("hashmap"),LRU,timeGPU,timeCPU);
			delete cloud;
			return 1;
                    } else if(argc == 9) {
			if (getBool(argv[6])) {
				PointCloud<std::unordered_map> * cloud = new PointCloud<std::unordered_map>(std::string(""),std::string(argv[2])+"/",atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),getBool(argv[7]));

				//double timeCPU = radiiEstimationCPU(*cloud);
				double timeCPU = getMaxLevelCPU(*cloud);
				//double timeGPU = radiiEstimationGPU(*cloud, devices);
				double timeGPU = getMaxLevelGPU(*cloud, devices);
				std::cout << "(Time CPU/GPU: " << timeCPU << "/" << timeGPU << " s.)" << std::endl;
				output(argv[8],cloud->getL1Stats(),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),std::string("hashmap"),getBool(argv[7]),timeGPU,timeCPU);
				delete cloud;
			} else {
				PointCloud<std::map> * cloud = new PointCloud<std::map>(std::string(""),std::string(argv[2])+"/",atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),getBool(argv[7]));
				//double timeCPU = radiiEstimationCPU(*cloud);
				double timeCPU = getMaxLevelCPU(*cloud);
				//double timeGPU = radiiEstimationGPU(*cloud, devices);
				double timeGPU = getMaxLevelGPU(*cloud, devices);
				std::cout << "(Time CPU/GPU: " << timeCPU << "/" << timeGPU << " s.)" << std::endl;
				output(argv[8],cloud->getL1Stats(),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),std::string("map"),getBool(argv[7]),timeGPU,timeCPU);
				delete cloud;
				}
			return 1;
                    } else {
			help();
			return 1;
                    }
                }

	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	//exit(1);

    return 0;

}

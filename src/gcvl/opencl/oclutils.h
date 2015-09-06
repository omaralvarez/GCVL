/*
 *
 * GPGPU Computer Vision Library (GCVL)
 *
 * Copyright (c) Nicolas Bigaouette 2011 <nbigaouette@gmail.com>
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

#include <string>
#include <list>
#include <map>
#include <climits>
#include <stdint.h>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#ifndef std_cout
#define std_cout std::cout
#include <iostream>
#include <fstream>
#include <fstream>
#endif // #ifndef std_cout

const std::string OPENCL_PLATFORMS_NVIDIA("nvidia");
const std::string OPENCL_PLATFORMS_AMD("amd");
const std::string OPENCL_PLATFORMS_INTEL("intel");
const std::string OPENCL_PLATFORMS_APPLE("apple");

// *****************************************************************************
#define OpenCL_Test_Success(err, fct_name)                          \
if ((err) != CL_SUCCESS)                                            \
{                                                                   \
    std_cout                                                        \
        << "ERROR calling " << fct_name << "() ("                   \
        << __FILE__ << " line " << __LINE__ << "): "                \
        << OpenCL_Error_to_String(err) << "\n" << std::flush;               \
    abort();                                                        \
}

// *****************************************************************************
#define OpenCL_Release_Kernel(err, opencl_kernel)                   \
{                                                                   \
    if ((opencl_kernel)) err = clReleaseKernel((opencl_kernel));    \
    OpenCL_Test_Success(err, "clReleaseKernel");                    \
}

// *****************************************************************************
#define OpenCL_Release_Program(err, opencl_program)                 \
{                                                                   \
    if ((opencl_program)) err = clReleaseProgram((opencl_program)); \
    OpenCL_Test_Success(err, "clReleaseProgram");                   \
}

// *****************************************************************************
#define OpenCL_Release_CommandQueue(err, opencl_cqueue)             \
{                                                                   \
    if ((opencl_cqueue)) err = clReleaseCommandQueue((opencl_cqueue));\
    OpenCL_Test_Success(err, "clReleaseCommandQueue");              \
}

// *****************************************************************************
#define OpenCL_Release_Memory(err, opencl_array)                    \
{                                                                   \
    if ((opencl_array)) err = clReleaseMemObject((opencl_array));   \
    OpenCL_Test_Success(err, "clReleaseMemObject");                 \
}

// *****************************************************************************
#define OpenCL_Release_Context(err, opencl_context)                 \
{                                                                   \
    if ((opencl_context)) err = clReleaseContext((opencl_context)); \
    OpenCL_Test_Success(err, "clReleaseContext");                   \
}

class OpenCL_platform;
class OpenCL_platforms_list;
class OpenCL_device;
class OpenCL_devices_list;
class OpenCL_Kernel;

// *****************************************************************************
// Nvidia extensions. On non-nvidia, needs to define those.
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#endif
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#endif
#ifndef CL_DEVICE_REGISTERS_PER_BLOCK_NV
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV            0x4002
#endif
#ifndef CL_DEVICE_WARP_SIZE_NV
#define CL_DEVICE_WARP_SIZE_NV                      0x4003
#endif
#ifndef CL_DEVICE_GPU_OVERLAP_NV
#define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
#endif
#ifndef CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
#endif
#ifndef CL_DEVICE_INTEGRATED_MEMORY_NV
#define CL_DEVICE_INTEGRATED_MEMORY_NV              0x4006
#endif

// *****************************************************************************
std::string OpenCL_Error_to_String(cl_int error);

// *****************************************************************************
bool Verify_if_Device_is_Used(const int device_id, const int platform_id_offset,
                              const std::string &platform_name, const std::string &device_name);

// *****************************************************************************
char *read_opencl_kernel(const std::string filename, int *length);

// *****************************************************************************
class OpenCL_device
{
    private:
        bool                            object_is_initialized;
        int                             device_id;
        cl_device_id                    device;
        cl_context                      context;
        bool                            device_is_gpu;
        bool                            device_is_in_use;

        // OpenCL device's information.
        // See http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
        cl_uint                         address_bits;
        cl_bool                         available;
        cl_bool                         compiler_available;
        cl_device_fp_config             double_fp_config;
        cl_bool                         endian_little;
        cl_bool                         error_correction_support;
        cl_device_exec_capabilities     execution_capabilities;
        cl_ulong                        global_mem_cache_size;
        cl_device_mem_cache_type        global_mem_cache_type;
        cl_uint                         global_mem_cacheline_size;
        cl_ulong                        global_mem_size;
        cl_device_fp_config             half_fp_config;
        cl_bool                         image_support;
        size_t                          image2d_max_height;
        size_t                          image2d_max_width;
        size_t                          image3d_max_depth;
        size_t                          image3d_max_height;
        size_t                          image3d_max_width;
        cl_ulong                        local_mem_size;
        cl_device_local_mem_type        local_mem_type;
        cl_uint                         max_clock_frequency;
        cl_uint                         max_compute_units;
        cl_uint                         max_constant_args;
        cl_ulong                        max_constant_buffer_size;
        cl_ulong                        max_mem_alloc_size;
        size_t                          max_parameter_size;
        cl_uint                         max_read_image_args;
        cl_uint                         max_samplers;
        size_t                          max_work_group_size;
        cl_uint                         max_work_item_dimensions;
        size_t                          max_work_item_sizes[3];
        cl_uint                         max_write_image_args;
        cl_uint                         mem_base_addr_align;
        cl_uint                         min_data_type_align_size;
        cl_platform_id                  platform;
        cl_uint                         preferred_vector_width_char;
        cl_uint                         preferred_vector_width_short;
        cl_uint                         preferred_vector_width_int;
        cl_uint                         preferred_vector_width_long;
        cl_uint                         preferred_vector_width_float;
        cl_uint                         preferred_vector_width_double;
        size_t                          profiling_timer_resolution;
        cl_command_queue_properties     queue_properties;
        cl_device_fp_config             single_fp_config;
        cl_device_type                  type;
        cl_uint                         vendor_id;

        std::string                     extensions;
        std::string                     name;
        std::string                     profile;
        std::string                     vendor;
        std::string                     version;
        std::string                     driver_version;
        std::string                     type_string;
        std::string                     queue_properties_string;
        std::string                     single_fp_config_string;

        // Nvidia specific extensions
        // http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/docs/OpenCL_Extensions/cl_nv_device_attribute_query.txt
        bool                            is_nvidia;
        cl_uint                         nvidia_device_compute_capability_major;
        cl_uint                         nvidia_device_compute_capability_minor;
        cl_uint                         nvidia_device_registers_per_block;
        cl_uint                         nvidia_device_warp_size;
        cl_bool                         nvidia_device_gpu_overlap;
        cl_bool                         nvidia_device_kernel_exec_timeout;
        cl_bool                         nvidia_device_integrated_memory;

        // A lock can be acquired on the device only if another program
        // did not acquired one before. If the program detects that the device
        // was is used by another process, it won't try to lock or unlock the device.
        bool                            is_lockable;
        bool                            file_locked;
        int                             lock_file;

    public:

        const OpenCL_platform          *parent_platform;

        OpenCL_device();
        ~OpenCL_device();
        void Destructor();
        //! Obtains the parent platform.
        /*!
          \return parent platform.
        */
        const OpenCL_platform *         Get_Parent_Platform()       { return parent_platform;   }
        //! Obtains the device name.
        /*!
          \return device name.
        */
        std::string                     Get_Name() const            { return name;              }
        //! Obtains the number of compute units of the device.
        /*!
          \return compute units available.
        */
        cl_uint                         Get_Compute_Units() const   { return max_compute_units; }
        //! Obtains the device id.
        /*!
          \return device id.
        */
        int                             Get_ID() const              { return device_id;         }
        //! Obtains the OpenCL device id.
        /*!
          \return OpenCL device id.
        */
        cl_device_id &                  Get_Device()                { return device;            }
        //! Obtains the OpenCL context of the device.
        /*!
          \return OpenCL device context.
        */
        cl_context &                    Get_Context()               { return context;           }
        //! Lets the user know if the device is in use.
        /*!
          \return device usage.
        */
        bool                            Is_In_Use()                 { return device_is_in_use;  }
        //! Checks if device is lockable.
        /*!
          \return device locking flag.
        */
        bool                            Is_Lockable()               { return is_lockable;       }
        //! Sets device locking capabilities.
        void                            Set_Lockable(const bool _is_lockable) { is_lockable = _is_lockable; }

        void                            Set_Information(const int _id, cl_device_id _device, const int platform_id_offset,
                                                        const std::string &platform_name, const bool _device_is_gpu,
                                                        const OpenCL_platform * const _parent_platform);

        cl_int                          Set_Context();
        void                            Print() const;
        void                            Lock();
        void                            Unlock();
        bool                            operator<(const OpenCL_device &b) const;
};

// *****************************************************************************
class OpenCL_devices_list
{
    private:
        bool                            is_initialized;
        const OpenCL_platform          *platform;
        std::list<OpenCL_device>        device_list;
        cl_uint                         nb_cpu;
        cl_uint                         nb_gpu;
        int                             err;
        bool                            are_all_devices_in_use;

    public:

        OpenCL_device                  *preferred_device;

        OpenCL_devices_list();
        ~OpenCL_devices_list();

        void                            Set_Preferred_OpenCL(const int _preferred_device = -1);
        OpenCL_device &                 Preferred_OpenCL();
        //! Obtains the preferred OpenCL device of the list.
        /*!
          \return preferred OpenCL device id.
        */
        cl_device_id &                  Preferred_OpenCL_Device()         { return Preferred_OpenCL().Get_Device(); }
        //! Obtains the preferred OpenCL device context.
        /*!
          \return preferred OpenCL device contex.
        */
        cl_context &                    Preferred_OpenCL_Device_Context() { return Preferred_OpenCL().Get_Context(); }
        //! Obtains the type of the devices present on list.
        /*!
          \return corresponding type key.
        */
        int                             nb_devices()                     { return nb_cpu + nb_gpu; }
        void                            Print() const;
        void                            Initialize(const OpenCL_platform &_platform,
                                                   const std::string &preferred_platform);

};

// *****************************************************************************
class OpenCL_platform
{
    private:
        cl_platform_id                  id;
        std::string                     profile;
        std::string                     version;
        std::string                     name;
        std::string                     vendor;
        std::string                     extensions;
        std::string                     key;
        OpenCL_platforms_list           *platform_list;
        int                             id_offset;
    public:
        OpenCL_devices_list             devices_list;
        OpenCL_platform();

        void                            Initialize(std::string _key, int id_offset, cl_platform_id _id,
                                                   OpenCL_platforms_list *_platform_list, const std::string preferred_platform);
        //! Obtains the preferred OpenCL device.
        /*!
          \return preferred OpenCL device.
        */
        OpenCL_device &                 Preferred_OpenCL()                   { return devices_list.Preferred_OpenCL(); }
        //! Obtains the preferred OpenCL device id.
        /*!
          \return preferred OpenCL device id.
        */
        cl_device_id &                  Preferred_OpenCL_Device()            { return devices_list.Preferred_OpenCL_Device(); }
        //! Obtains the preferred OpenCL device context.
        /*!
          \return preferred OpenCL device context.
        */
        cl_context &                    Preferred_OpenCL_Device_Context()    { return devices_list.Preferred_OpenCL_Device_Context(); }
        //! Obtains the parent OpenCL platform list.
        /*!
          \return parent OpenCL platform list.
        */
        OpenCL_platforms_list *         Platform_List() const               { return platform_list; }
        void                            Print_Preferred() const;
        //! Obtains the platform key.
        /*!
          \return platform key.
        */
        std::string                     Key() const                         { return key; }
        //! Obtains the platform name.
        /*!
          \return platform name.
        */
        std::string   const             Name() const                        { return name; }
        //! Obtains the platform id.
        /*!
          \return platform id.
        */
        cl_platform_id                  Id() const                          { return id; }
        //! Obtains the platform id offset.
        /*!
          \return platform id offset.
        */
        int                             Id_Offset() const                   { return id_offset; }
        void                            Lock_Best_Device();
        void                            Print() const;
};

// *****************************************************************************
class OpenCL_platforms_list
{
    private:
        std::map<std::string,OpenCL_platform>   platforms;
        std::string                     preferred_platform;
        bool                            use_locking;
    public:
        void                            Initialize(const std::string &_preferred_platform, const bool _use_locking = true);
        void                            Print() const;
        void                            Print_Preferred() const;
        //! Obtains the active platform.
        /*!
          \return preferred platform.
        */
        std::string                     Get_Running_Platform()              { return preferred_platform; }
        //! Function that obtains locking status.
        /*!
          \return locking status.
        */
        bool                            Use_Locking() const                 { return use_locking; }

        OpenCL_platform & operator[](const std::string key);
        void                            Set_Preferred_OpenCL(const int _preferred_device = -1);
};

// **************************************************************
class OpenCL_Kernel
{
    public:

        OpenCL_Kernel();
        OpenCL_Kernel(std::string _filename, const cl_context &_context,
                      const cl_device_id &_device_id);
        ~OpenCL_Kernel();
        void Initialize(std::string _filename, const cl_context &_context,
                        const cl_device_id &_device_id);

        void Build(std::string _kernel_name);

        // By default global_y is one, local_x is MAX_WORK_SIZE and local_y is one.
        void Compute_Work_Size(size_t _global_x, size_t _global_y, size_t _local_x, size_t _local_y);

        const cl_kernel& Get_Kernel() const;

        size_t *Get_Global_Work_Size() const;
        size_t *Get_Local_Work_Size() const;

        int Get_Dimension() const;
        void Append_Compiler_Option(const std::string option);

        void Launch(const cl_command_queue &command_queue);

        static int Get_Multiple(int n, int base);

    private:

        std::string filename;
        cl_context context;
        cl_device_id device_id;

        std::string compiler_options;
        std::string kernel_name;

        int dimension;
        int p;
        int q;

        cl_program program;

        cl_kernel kernel;
        size_t *global_work_size;
        size_t *local_work_size;

        // Debugging variables
        cl_int err;
        cl_event event;

        void Load_Program_From_File();

        void Build_Executable(const bool verbose = true);
};


// *****************************************************************************
template <class T>
class OpenCL_Array
{
private:
    bool array_is_padded;               // Will the array need to be padded for checksumming?
    int N;                              // Number of elements in array
    size_t sizeof_element;              // Size of each array elements
    uint64_t new_array_size_bytes;      // Size (bytes) of new padded array
    T     *host_array;                  // Pointer to start of host array, INCLUDIǸG padding
    uint64_t nb_1024bits_blocks;        // Number of 1024 bits blocks in padded array
    std::string platform;               // OpenCL platform
    cl_context context;                 // OpenCL context
    cl_command_queue command_queue;     // OpenCL command queue
    cl_device_id device;                // OpenCL device
    cl_int err;                         // Error code

    uint8_t host_checksum[64];          // SHA512 checksum on host memory (512 bits)
    uint8_t device_checksum[64];        // SHA512 checksum on device memory (512 bits)
    static const int buff_size_checksum = sizeof(uint8_t) * 64;

    OpenCL_Kernel kernel_checksum;      // Kernel for checksum calculation

    // Allocated memory on device
    cl_mem device_array;                // Memory of device
    cl_mem cl_array_size_bit;
    cl_mem cl_sha512sum;

public:
    OpenCL_Array();
    void Initialize(int _N, const size_t _sizeof_element,
                    T *&host_array,
                    const cl_context &_context, cl_mem_flags flags,
                    std::string _platform,
                    const cl_command_queue &_command_queue,
                    const cl_device_id &_device,
                    const bool _checksum_array);
    void Release_Memory();
    void Host_to_Device();
    void Device_to_Host();
    std::string Host_Checksum();
    std::string Device_Checksum();
    void Validate_Data();

    //! Obtains the device array OpenCL memory pointer.
    /*!
      \return OpenCL memory pointer.
    */
    inline cl_mem * Get_Device_Array() { return &device_array; }
    //! Obtains the host array memory pointer.
    /*!
      \return memory pointer.
    */
    inline T *      Get_Host_Pointer() { return  host_array;   }
    void Set_as_Kernel_Argument(cl_kernel &kernel, const int order);
};

// *****************************************************************************
template <class T>
class OpenCL_Data
{
private:
    T data;
    cl_int err;
public:
    OpenCL_Data();
    void Initialize(T _data);
    void Set_as_Kernel_Argument(cl_kernel &kernel, const int order);
};

// ********** End of file ***************************************

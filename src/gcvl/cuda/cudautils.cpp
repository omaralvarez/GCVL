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

#include "cudautils.h"
#include "../gcvlutils.h"

#include <cassert>
#include <algorithm>

#ifdef __DRIVER_TYPES_H__
static const std::string CUDA_Error_to_String(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
        case cudaErrorOperatingSystem:
            return "cudaErrorOperatingSystem";

        case cudaErrorPeerAccessUnsupported:
            return "cudaErrorPeerAccessUnsupported";

        case cudaErrorLaunchMaxDepthExceeded:
            return "cudaErrorLaunchMaxDepthExceeded";

        case cudaErrorLaunchFileScopedTex:
            return "cudaErrorLaunchFileScopedTex";

        case cudaErrorLaunchFileScopedSurf:
            return "cudaErrorLaunchFileScopedSurf";

        case cudaErrorSyncDepthExceeded:
            return "cudaErrorSyncDepthExceeded";

        case cudaErrorLaunchPendingCountExceeded:
            return "cudaErrorLaunchPendingCountExceeded";

        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";

        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";

        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";

        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";

        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";

        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";

        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx";

        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext";

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static const std::string CUDA_Error_to_String(CUresult error)
{
    switch (error)
    {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";

        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";

        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";

        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";

        case CUDA_ERROR_DEINITIALIZED:
            return "CUDA_ERROR_DEINITIALIZED";

        case CUDA_ERROR_PROFILER_DISABLED:
            return "CUDA_ERROR_PROFILER_DISABLED";

        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

        case CUDA_ERROR_NO_DEVICE:
            return "CUDA_ERROR_NO_DEVICE";

        case CUDA_ERROR_INVALID_DEVICE:
            return "CUDA_ERROR_INVALID_DEVICE";

        case CUDA_ERROR_INVALID_IMAGE:
            return "CUDA_ERROR_INVALID_IMAGE";

        case CUDA_ERROR_INVALID_CONTEXT:
            return "CUDA_ERROR_INVALID_CONTEXT";

        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

        case CUDA_ERROR_MAP_FAILED:
            return "CUDA_ERROR_MAP_FAILED";

        case CUDA_ERROR_UNMAP_FAILED:
            return "CUDA_ERROR_UNMAP_FAILED";

        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return "CUDA_ERROR_ARRAY_IS_MAPPED";

        case CUDA_ERROR_ALREADY_MAPPED:
            return "CUDA_ERROR_ALREADY_MAPPED";

        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return "CUDA_ERROR_NO_BINARY_FOR_GPU";

        case CUDA_ERROR_ALREADY_ACQUIRED:
            return "CUDA_ERROR_ALREADY_ACQUIRED";

        case CUDA_ERROR_NOT_MAPPED:
            return "CUDA_ERROR_NOT_MAPPED";

        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return "CUDA_ERROR_ECC_UNCORRECTABLE";

        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return "CUDA_ERROR_UNSUPPORTED_LIMIT";

        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";

        case CUDA_ERROR_INVALID_PTX:
            return "CUDA_ERROR_INVALID_PTX";

        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";

        case CUDA_ERROR_INVALID_SOURCE:
            return "CUDA_ERROR_INVALID_SOURCE";

        case CUDA_ERROR_FILE_NOT_FOUND:
            return "CUDA_ERROR_FILE_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

        case CUDA_ERROR_OPERATING_SYSTEM:
            return "CUDA_ERROR_OPERATING_SYSTEM";

        case CUDA_ERROR_INVALID_HANDLE:
            return "CUDA_ERROR_INVALID_HANDLE";

        case CUDA_ERROR_NOT_FOUND:
            return "CUDA_ERROR_NOT_FOUND";

        case CUDA_ERROR_NOT_READY:
            return "CUDA_ERROR_NOT_READY";

        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return "CUDA_ERROR_ILLEGAL_ADDRESS";

        case CUDA_ERROR_LAUNCH_FAILED:
            return "CUDA_ERROR_LAUNCH_FAILED";

        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return "CUDA_ERROR_LAUNCH_TIMEOUT";

        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

        case CUDA_ERROR_ASSERT:
            return "CUDA_ERROR_ASSERT";

        case CUDA_ERROR_TOO_MANY_PEERS:
            return "CUDA_ERROR_TOO_MANY_PEERS";

        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

        case CUDA_ERROR_HARDWARE_STACK_ERROR:
            return "CUDA_ERROR_HARDWARE_STACK_ERROR";

        case CUDA_ERROR_ILLEGAL_INSTRUCTION:
            return "CUDA_ERROR_ILLEGAL_INSTRUCTION";

        case CUDA_ERROR_MISALIGNED_ADDRESS:
            return "CUDA_ERROR_MISALIGNED_ADDRESS";

        case CUDA_ERROR_INVALID_ADDRESS_SPACE:
            return "CUDA_ERROR_INVALID_ADDRESS_SPACE";

        case CUDA_ERROR_INVALID_PC:
            return "CUDA_ERROR_INVALID_PC";

        case CUDA_ERROR_NOT_PERMITTED:
            return "CUDA_ERROR_NOT_PERMITTED";

        case CUDA_ERROR_NOT_SUPPORTED:
            return "CUDA_ERROR_NOT_SUPPORTED";

        case CUDA_ERROR_UNKNOWN:
            return "CUDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}
#endif

CUDA_device::CUDA_device(int _id) : id(_id) {

	//TODO Check errors
	err = cudaGetDeviceProperties(&properties, id);
	CUDA_Test_Success(err,"cudaGetDeviceProperties()");

	compute = false;
	sm_per_multiproc = 0;
	compute_units = 0;

	// Check if this GPU is not running on Compute Mode prohibited
    if (properties.computeMode != cudaComputeModeProhibited) {
        if (properties.major > 0 && properties.major < 9999) {
            sm_per_multiproc = SMVer2CU(properties.major, properties.minor);
			compute_units = sm_per_multiproc * properties.multiProcessorCount;
			compute_perf = (unsigned long long) properties.multiProcessorCount * sm_per_multiproc * properties.clockRate;
			compute = true;
        }
	}

}

bool CUDA_device::operator<(const CUDA_device &other) const {

    bool result = false;

	if(this->properties.major > other.properties.major) // "this" wins (having higher arch).
		result = true;
	else if(other.properties.major > this->properties.major) // "other" wins (having higher arch).
		result = false;
    else if (this->compute_perf > other.compute_perf) // **For same arch**: "this" wins (having more compute units).
        result = true;
    else                                                 // "other" wins (having more or equal compute units).
        result = false;

    return result;

}

void CUDA_device::Set_Information(unsigned long long _compute_perf, int arch) {

	compute_perf = _compute_perf;
	properties.major = arch;

}

void CUDA_device::Print() const {

	std::cout << "    "; Print_N_Times("-", 105);

    std::cout
        << "    name: " << properties.name << std::endl
		<< "        id:                             " << id << std::endl
		<< "        Compute prohibited:             " << (compute ? "No" : "Yes") << std::endl
		<< "        Major revision number:          " << properties.major << std::endl
		<< "        Minor revision number:          " << properties.minor << std::endl
		<< "        Global memory:                  " << properties.totalGlobalMem << " bytes" << std::endl
		<< "        CUDA asynchronous engines:      " << properties.asyncEngineCount << std::endl
		<< "        Number of multiprocessors:      " << properties.multiProcessorCount << std::endl
		<< "        SMs per multiprocessor:         " << sm_per_multiproc << std::endl
		<< "        Number of compute units:        " << compute_units << std::endl
		<< "        Constant memory:                " << properties.totalConstMem << " bytes" << std::endl
		<< "        Shared memory per block:        " << properties.sharedMemPerBlock << " bytes" << std::endl
		<< "        Registers available per block:  " << properties.regsPerBlock << std::endl
		<< "        Warp size:                      " << properties.warpSize << std::endl
		<< "        Max. threads per block:         " << properties.maxThreadsPerBlock << std::endl
		<< "        Max. block dimension size:      " << properties.maxThreadsDim[0] << " " << properties.maxThreadsDim[2] << " " << properties.maxThreadsDim[2] << std::endl
		<< "        Max. grid dimension size:       " << properties.maxGridSize[0] << " " << properties.maxGridSize[2] << " " << properties.maxGridSize[2] << std::endl
		<< "        Maximum memory pitch:           " << properties.memPitch << " bytes" << std::endl
		<< "        Texture alignment:              " << properties.textureAlignment << " bytes" << std::endl
		<< "        Clock rate:                     " << properties.clockRate * 1e-6f << " GHz" << std::endl;

}

CUDA_devices_list::CUDA_devices_list() : is_initialized(false), preferred_device(-1) { }

void CUDA_devices_list::Initialize() {

	err = cudaGetDeviceCount(&count);
	CUDA_Test_Success(err,"cudaGetDeviceCount()");

	device_list.reserve(count);

	int i = 0;
	std::generate_n(std::back_inserter(device_list), count, [i] () mutable { return CUDA_device(i++); });

	std::sort(device_list.begin(), device_list.end());

	preferred_device = device_list.front().Get_ID();

	err = cudaSetDevice(preferred_device);
	CUDA_Test_Success(err,"cudaSetDevice()");

	is_initialized = true;

}

void CUDA_devices_list::Print() {

	if (!device_list.size()) {
        std::cout << "        None" << std::endl;
    } else {
        for (std::vector<CUDA_device>::const_iterator it = device_list.begin() ; it != device_list.end() ; ++it)
            it->Print();

        std::cout << "        "; Print_N_Times("*", 101);
        std::cout << "        Order of preference for CUDA devices for this platform:\n";
        int i = 0;
        for (std::vector<CUDA_device>::const_iterator it = device_list.begin() ; it != device_list.end() ; ++it)
        {
            std::cout << "        " << i++ << ".   " << it->Get_Name() << " (id = " << it->Get_ID() << ")\n";
        }
        std::cout << "        "; Print_N_Times("*", 101);
    }

}
//CUDA_de::CUDA_Array()

template <class T>
CUDA_Array<T>::CUDA_Array()
{
    N                           = 0;
    sizeof_element              = 0;
    new_array_size_bytes        = 0;
    host_array                  = NULL;
	device_array                = NULL;
}

template <class T>
void CUDA_Array<T>::Initialize(int _N, T *&_host_array)
{
    assert(_host_array != NULL);

    N               = _N;
    sizeof_element  = sizeof(T);
    host_array      = _host_array;
    new_array_size_bytes = N * sizeof_element;

	err = cudaMalloc(&device_array, new_array_size_bytes);
	CUDA_Test_Success(err,"cudaMalloc()");

}

template <class T>
void CUDA_Array<T>::Release_Memory()
{
    if (device_array)
        cudaFree(device_array);
}

template <class T>
void CUDA_Array<T>::Host_to_Device()
{
	err = cudaMemcpy(device_array, host_array, new_array_size_bytes, cudaMemcpyHostToDevice);
	CUDA_Test_Success(err,"cudaMemcpy()");
}

// *****************************************************************************
template <class T>
void CUDA_Array<T>::Device_to_Host()
{
	err = cudaMemcpy(host_array, device_array, new_array_size_bytes, cudaMemcpyDeviceToHost);
	CUDA_Test_Success(err,"cudaMemcpy()");
}

//TODO Add more types?
template class CUDA_Array<float>;
template class CUDA_Array<double>;
template class CUDA_Array<int>;
template class CUDA_Array<unsigned int>;
template class CUDA_Array<char>;
template class CUDA_Array<unsigned char>;

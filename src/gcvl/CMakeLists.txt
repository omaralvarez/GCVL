project(gcvl-lib)

SET(SOURCE
	algorithm.cpp
	blockmatching.cpp
)

SET(HEADERS
	gcvlutils.h
	gcvlconfig.h
	export.h
	algorithm.h
	blockmatching.h
)

include_directories (${CMAKE_SOURCE_DIR})

if(BUILD_OPENCL)

	list(APPEND SOURCE
		opencl/oclutils.cpp
		opencl/oclcore.cpp
		opencl/oclalgorithm.cpp
		opencl/oclblockmatching.cpp
	)

	list(APPEND HEADERS
		opencl/oclutils.h
		opencl/oclcore.h
		opencl/oclalgorithm.h
		opencl/oclblockmatching.h
	)

endif(BUILD_OPENCL)

if(BUILD_CUDA)

	# Target older architectures for those who haven't upgraded their graphics card to the latest models.
	#set(CUDA_NVCC_FLAGS "-gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 ")

	list(APPEND SOURCE
		cuda/cudacore.cpp
		cuda/cudautils.cpp
	)

	list(APPEND HEADERS
		cuda/cudacore.h
		cuda/cudautils.h
		cuda/kernels/test.h
	)
	
	SET(CUDA_KERNELS
		cuda/kernels/test.cu
	)
	
	cuda_add_library(gcvl_kernels ${CUDA_KERNELS})

endif(BUILD_CUDA)

add_library(gcvl SHARED ${SOURCE} ${HEADERS})

IF(BUILD_OPENCL)
	target_link_libraries(gcvl ${OPENCL_LIBRARIES})
ENDIF(BUILD_OPENCL)

IF(BUILD_CUDA)
	target_link_libraries(gcvl gcvl_kernels ${CUDA_LIBRARIES})
ENDIF(BUILD_CUDA)

IF (MSVC OR APPLE)
target_link_libraries(gcvl ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
ELSE (MSVC OR APPLE)
target_link_libraries(gcvl ${Boost_LIBRARIES} ${OpenCV_LIBS})
ENDIF ()


IF (MSVC)
  set_target_properties (gcvl PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
ENDIF ()

IF(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
ENDIF()

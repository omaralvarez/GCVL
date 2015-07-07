project(gcvl-lib)

SET(SOURCE
	algorithm.cpp
	blockmatching.cpp
)

SET(HEADERS
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

	list(APPEND SOURCE
		cuda/cudacore.cpp
	)

	list(APPEND HEADERS
		cuda/cudacore.h
	)

endif(BUILD_CUDA)

add_library(gcvl SHARED ${SOURCE} ${HEADERS})

IF(BUILD_OPENCL)
	target_link_libraries(gcvl ${OPENCL_LIBRARIES})
ENDIF(BUILD_OPENCL)

IF(BUILD_CUDA)
	target_link_libraries(gcvl ${CUDA_LIBRARIES})
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

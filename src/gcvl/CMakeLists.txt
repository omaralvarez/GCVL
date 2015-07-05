project(gcvl-lib)

SET(SOURCE
	algorithm.cpp
)

SET(HEADERS
	gcvlconfig.h
	export.h
	algorithm.h
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

	#include_directories (${CMAKE_SOURCE_DIR}/opencl)

endif(BUILD_OPENCL)

add_library(gcvl SHARED ${SOURCE} ${HEADERS})
IF (MSVC OR APPLE)
target_link_libraries(gcvl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
ELSE (MSVC OR APPLE)
target_link_libraries(gcvl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
ENDIF ()

IF (MSVC)
  set_target_properties (gcvl PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
ENDIF ()

IF(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
ENDIF()

project(gcvl-lib)

SET(SOURCE
	oclutils.cpp
	oclalgorithm.cpp
	oclblockmatching.cpp
)

SET(HEADERS
	oclutils.h
	oclalgorithm.h
	oclblockmatching.h
	gcvlconfig.h
	export.h
)
include_directories (${CMAKE_SOURCE_DIR}) 

add_library(gcvl SHARED ${SOURCE} ${HEADERS})
target_link_libraries(gcvl ${OPENCL_LIBRARIES})

IF (MSVC)
  set_target_properties (gcvl PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
ENDIF ()
IF(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
ENDIF()
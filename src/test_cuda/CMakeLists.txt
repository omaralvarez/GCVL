project(test_cuda)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(test_cuda test_cuda.cpp)
IF (MSVC OR APPLE)
	target_link_libraries(test_cuda gcvl ${OpenCV_LIBRARIES})
ELSE (MSVC OR APPLE)
	target_link_libraries(test_cuda gcvl ${OpenCV_LIBS})
ENDIF ()


IF(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
ENDIF()

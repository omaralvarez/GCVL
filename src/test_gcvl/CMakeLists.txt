project(test_gcvl)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(test_gcvl test_gcvl.cpp)
target_link_libraries(test_gcvl gcvl)

IF(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
ENDIF()


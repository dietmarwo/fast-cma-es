#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_COMPILER             "/usr/bin/clang++")

INCLUDE_DIRECTORIES(/pathToArmadillo/armadillo-9.800.3/)

#set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
endif()

PROJECT(acmalib)

add_library(acmalib SHARED acmaesoptimizer.cpp)
target_link_libraries(acmalib armadillo mkl_rt)   

set(CMAKE_INSTALL_LIBDIR ${CMAKE_BINARY_DIR}/../fcmaes)

install(TARGETS acmalib EXPORT ArmadilloLibraryDepends
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
 

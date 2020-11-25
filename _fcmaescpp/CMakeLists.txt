#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

INCLUDE_DIRECTORIES(/home/xxx/ctoc/fcmaes/_fcmaescpp/include)

#set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
endif()

PROJECT(acmalib)

add_library(acmalib SHARED acmaesoptimizer.cpp deoptimizer.cpp daoptimizer.cpp hawksoptimizer.cpp gcldeoptimizer.cpp lcldeoptimizer.cpp ldeoptimizer.cpp biteoptimizer.cpp csmaoptimizer.cpp ascent.cpp)

set(CMAKE_INSTALL_LIBDIR ${CMAKE_BINARY_DIR}/../fcmaes/lib)

install(TARGETS acmalib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
 

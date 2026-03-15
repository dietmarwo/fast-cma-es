cmake_minimum_required(VERSION 3.18...3.31)

project(fcmaes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_PACKAGE_REGISTRY OFF CACHE BOOL
  "Do not write package entries to the user registry while configuring." FORCE)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE BOOL
  "Disable package registry export to keep container builds self-contained." FORCE)

option(FCMAES_FETCH_NANOBIND "Fetch nanobind if it is not installed in the active build environment." OFF)
option(FCMAES_FETCH_EIGEN "Fetch Eigen if it is not available on the local system." OFF)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE FCMAES_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE FCMAES_NANOBIND_CMAKE_RESULT
  ERROR_QUIET
)

if(FCMAES_NANOBIND_CMAKE_RESULT EQUAL 0 AND FCMAES_NANOBIND_CMAKE_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${FCMAES_NANOBIND_CMAKE_DIR}")
endif()

find_package(nanobind CONFIG QUIET)

if(NOT nanobind_FOUND)
  if(FCMAES_FETCH_NANOBIND)
    include(FetchContent)
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(nanobind)
  else()
    message(FATAL_ERROR
      "nanobind was not found. Install it into the active Python environment "
      "or reconfigure with -DFCMAES_FETCH_NANOBIND=ON.")
  endif()
endif()

find_package(Eigen3 3.4 CONFIG QUIET)

if(NOT Eigen3_FOUND)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(EIGEN3 QUIET IMPORTED_TARGET eigen3)
  endif()

  if(TARGET PkgConfig::EIGEN3)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_link_libraries(Eigen3::Eigen INTERFACE PkgConfig::EIGEN3)
  elseif(FCMAES_FETCH_EIGEN)
    include(FetchContent)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(eigen)
    if(NOT TARGET Eigen3::Eigen)
      if(TARGET eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
      else()
        message(FATAL_ERROR
          "Eigen was fetched but did not provide the expected Eigen targets.")
      endif()
    endif()
  else()
    message(FATAL_ERROR
      "Eigen3 was not found. Install Eigen3 or reconfigure with "
      "-DFCMAES_FETCH_EIGEN=ON.")
  endif()
endif()

add_library(fcmaes_backend STATIC
  _fcmaescpp/acmaesoptimizer.cpp
  _fcmaescpp/biteoptimizer.cpp
  _fcmaescpp/crfmnes.cpp
  _fcmaescpp/daoptimizer.cpp
  _fcmaescpp/deoptimizer.cpp
  _fcmaescpp/gtop.cpp
  _fcmaescpp/modeoptimizer.cpp
  _fcmaescpp/pgpe.cpp
)

set_target_properties(fcmaes_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fcmaes_backend PUBLIC Eigen3::Eigen)
target_include_directories(fcmaes_backend PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/_fcmaescpp/include"
)

nanobind_add_module(_fcmaes_ext NOMINSIZE
  _fcmaescpp/bindings/module.cpp
  _fcmaescpp/bindings/acma_bindings.cpp
  _fcmaescpp/bindings/bite_bindings.cpp
  _fcmaescpp/bindings/crfmnes_bindings.cpp
  _fcmaescpp/bindings/da_bindings.cpp
  _fcmaescpp/bindings/de_bindings.cpp
  _fcmaescpp/bindings/gtop_bindings.cpp
  _fcmaescpp/bindings/mode_bindings.cpp
  _fcmaescpp/bindings/pgpe_bindings.cpp
)

target_link_libraries(_fcmaes_ext PRIVATE Eigen3::Eigen fcmaes_backend)
target_include_directories(_fcmaes_ext PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/_fcmaescpp/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/_fcmaescpp/bindings"
)

install(
  TARGETS _fcmaes_ext
  LIBRARY DESTINATION fcmaes
  RUNTIME DESTINATION fcmaes
)

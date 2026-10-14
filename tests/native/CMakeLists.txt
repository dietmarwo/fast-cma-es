# Standalone tests for the header-only Pareto archive (no Python needed):
#   cmake -S tests/native -B build-native -DCMAKE_BUILD_TYPE=Debug -DFCMAES_SANITIZE=ON
#   cmake --build build-native && ctest --test-dir build-native --output-on-failure
cmake_minimum_required(VERSION 3.15)
project(fcmaes_native_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(FCMAES_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(FCMAES_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../_fcmaescpp/include)
enable_testing()

foreach(name test_pareto_kernel test_pareto_safety test_pareto_wrapper benchmark_pareto)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${FCMAES_INCLUDE})
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    if(FCMAES_SANITIZE)
      target_compile_options(${name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
                                             -fno-sanitize-recover=all)
      target_link_options(${name} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endforeach()

add_test(NAME pareto_kernel COMMAND test_pareto_kernel)
add_test(NAME pareto_safety COMMAND test_pareto_safety)
add_test(NAME pareto_wrapper COMMAND test_pareto_wrapper)
add_test(NAME pareto_benchmark_smoke COMMAND benchmark_pareto 20000)

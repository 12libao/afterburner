find_package(Kokkos REQUIRED)
cmake_minimum_required (VERSION 3.10)
project (testfly)
include(../../common.cmake)

# find_package(Kokkos REQUIRED)
add_executable(testfly testfly.cpp)
target_link_libraries(testfly Kokkos::kokkos)



cmake_minimum_required( VERSION 2.8.6 FATAL_ERROR )

# We want C++11
SET(GCC_FLAGS "-Wall -Wextra -Wpedantic" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS} -std=c++11 -DHAVE_GLOG" CACHE STRING "Flags used by the compiler during all build types")

project( learning-with-label-proportions )

add_definitions(-DLLP_VERSION="1")

option( BUILD_TOOLS "Build tools" OFF )
option( BUILD_TESTING "Build tests" ON )

find_package(Eigen3 REQUIRED)
include_directories( SYSTEM ${EIGEN3_INCLUDE_DIR} )

# Help cmake find itk
#find_package( Ceres REQUIRED PATHS "External/" )
#include_directories( ${CERES_INCLUDE_DIRS} )


# Set variable for libcmaes
set( CMAES_LIBRARIES ${CMAKE_SOURCE_DIR}/external/lib/libcmaes.so.0 )
include_directories( SYSTEM external/include )

include_directories( "/home/silas/Projects/bagged-data/include/" )

include_directories( include )
include_directories( include/llp )

#add_subdirectory( src )

if( BUILD_TOOLS )
  add_subdirectory( tools )
endif( BUILD_TOOLS )

if( BUILD_TESTING )
  enable_testing()
  add_subdirectory( test )
endif( BUILD_TESTING )



# include_directories( /usr/include/hdf5/serial )
# #include_directories( /usr/include/hdf5/mpi )

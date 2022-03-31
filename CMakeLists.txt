cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0074 NEW)

project(libparanumal VERSION 0.4.0)

enable_language(C CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

option(EXTERNAL_OCCA "Use an external OCCA build" OFF)
if(EXTERNAL_OCCA)
  message(STATUS "Using an external OCCA build.")
  find_package(occa REQUIRED)
  set(OCCA_TARGET OCCA::libocca)
else()
  message(FATAL_ERROR "Currently only external OCCA builds are enabled :(")
endif()

add_compile_definitions(
    "USE_OCCA_MEM_BYTE_ALIGN=64"
    "dfloat=double"
    "dfloatString=\"double\""
    "dfloatFormat=\"%lf\""
    "MPI_DFLOAT=MPI_DOUBLE"
    "dlong=int"
    "dlongString=\"int\""
    "dlongFormat=\"%d\""
    "MPI_DLONG=MPI_INT"
    "hlong=long long int"
    "hlongString=\"long long int\""
    "hlongFormat=\"%lld\""
    "MPI_HLONG=MPI_LONG_LONG_INT"
)

add_subdirectory(3rdParty)
add_subdirectory(libs)

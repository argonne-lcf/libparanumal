project(corelib)

set(corelib-src
  factor.cpp
  matrixConditionNumber.cpp
  matrixEig.cpp
  matrixInverse.cpp
  matrixRightSolve.cpp
  matrixTranspose.cpp
  parallelSort.cpp
  platformBuildKernel.cpp
  platformDeviceConfig.cpp
  platformProperties.cpp
  platformSettings.cpp
  settings.cpp  
)

add_library(core STATIC ${corelib-src})
target_include_directories(core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(core 
  PUBLIC 
    OCCA::libocca 
    MPI::MPI_CXX 
    BLAS::BLAS
    LAPACK::LAPACK
)

set_target_properties(core PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/core.hpp)

target_compile_definitions(core PUBLIC "LIBP_DIR=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS core)

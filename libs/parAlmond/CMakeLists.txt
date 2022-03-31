project(parAlmondlib)

set(parAlmondlib-src
  parAlmond.cpp
  parAlmondAMGLevel.cpp
  parAlmondAMGSetup.cpp
  parAlmondAMGSmoother.cpp
  parAlmondCoarseExact.cpp
  parAlmondCoarseOAS.cpp
  parAlmondCoarsenLevel.cpp
  parAlmondFormAggregates.cpp
  parAlmondGalerkinProd.cpp
  parAlmondKcycle.cpp
  parAlmondKernels.cpp
  parAlmondMultigrid.cpp
  parAlmondSettings.cpp
  parAlmondSmoothPrologator.cpp
  parAlmondSpMM.cpp
  parAlmondStrongGraph.cpp
  parAlmondTentativeProlongator.cpp
  parAlmondTranspose.cpp
  parAlmondVcycle.cpp
  parAlmondparCSR.cpp
)

add_library(parAlmond STATIC ${parAlmondlib-src})
target_include_directories(parAlmond
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(parAlmond PUBLIC linearSolver ogs OCCA::libocca MPI::MPI_CXX)
set_target_properties(parAlmond PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/parAlmond.hpp)

target_compile_definitions(parAlmond PUBLIC "PARALMOND_DIR=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS parAlmond)
install(DIRECTORY okl/ DESTINATION okl)

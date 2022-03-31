project(ogslib)

set(ogslib-src
  gs.cpp
  hostGather.cpp
  hostGatherScatter.cpp
  hostScatter.cpp
  occaGather.cpp
  occaGatheredHaloExchange.cpp
  occaGatherScatter.cpp
  occaScatter.cpp
  ogs.cpp
  ogsKernels.cpp
  ogsSetup.cpp
)

add_library(ogs STATIC ${ogslib-src})

target_include_directories(ogs 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(ogs PUBLIC OCCA::libocca gs MPI::MPI_CXX)
set_target_properties(ogs PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/ogs.hpp)

target_compile_definitions(ogs 
  PUBLIC 
    "USE_NEW_PINNED_MALLOC=1"
    "LIBP_DIR=\"${CMAKE_INSTALL_PREFIX}\""
    "OGS_DIR=\"${CMAKE_INSTALL_PREFIX}\""
)

install(TARGETS ogs)
install(DIRECTORY okl/ DESTINATION okl)
project(linAlglib)

set(linAlglib-src
  linAlg.cpp
  linAlgSetup.cpp
)

add_library(linAlg STATIC ${linAlglib-src})
target_include_directories(linAlg
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(linAlg PUBLIC core OCCA::libocca MPI::MPI_CXX)
set_target_properties(linAlg PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/linAlg.hpp)

target_compile_definitions(linAlg PUBLIC "LINALG_DIR=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS linAlg)
install(DIRECTORY okl/ DESTINATION okl)

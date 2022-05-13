project(cnsbench)

add_executable(cnsbench
  cnsBench.cpp
  src/cnsSettings.cpp
)

target_include_directories(cnsbench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cnsbench
  PUBLIC
    timeStepper
    mesh
    ogs
    linAlg
    core
    OCCA::libocca
    MPI::MPI_CXX
)

set_target_properties(cnsbench PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/cns.hpp)

target_compile_definitions(cnsbench PUBLIC "LIBP_DIR=\"${CMAKE_INSTALL_PREFIX}\"")
# target_compile_definitions(cnsbench PUBLIC "DCNS=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS cnsbench)
install(DIRECTORY okl/ DESTINATION okl)

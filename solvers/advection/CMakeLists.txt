project(advection)

add_executable(advection
  advectionMain.cpp
  src/advectionPlotFields.cpp
  src/advectionReport.cpp
  src/advectionRun.cpp
  src/advectionSettings.cpp
  src/advectionSetup.cpp
  src/advectionStep.cpp
)

target_include_directories(advection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(advection 
  PUBLIC
    timeStepper
    mesh
    ogs
    linAlg
    core
    OCCA::libocca
    MPI::MPI_CXX
)

target_compile_definitions(advection PUBLIC "LIBP_DIR=\"${CMAKE_INSTALL_PREFIX}\"")
target_compile_definitions(advection PUBLIC "DADVECTION=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS advection)
install(DIRECTORY okl/ DESTINATION okl)
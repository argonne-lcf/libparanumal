project(linearSolverlib)

set(linearSolverlib-src
  initialGuess.cpp
  linearSolver.cpp
  linearSolverNBFPCG.cpp
  linearSolverNBPCG.cpp
  linearSolverPCG.cpp
  linearSolverPGMRES.cpp
  linearSolverPMINRES.cpp
)

add_library(linearSolver STATIC ${linearSolverlib-src})
target_include_directories(linearSolver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(linearSolver PUBLIC core OCCA::libocca MPI::MPI_CXX)
set_target_properties(linearSolver PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/linearSolver.hpp)

target_compile_definitions(linearSolver PUBLIC "LINEARSOLVER_DIR=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS linearSolver)

install(DIRECTORY okl/ DESTINATION okl)

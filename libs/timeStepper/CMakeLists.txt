project(timeStepperlib)

set(timeStepperlib-src
  timeStepperAB3.cpp
  timeStepperDOPRI5.cpp
  timeStepperEXTBDF3.cpp
  timeStepperLSERK4.cpp
  timeStepperMRAB3.cpp
  timeStepperMRSAAB3.cpp
  timeStepperSAAB3.cpp
  timeStepperSARK4.cpp
  timeStepperSARK5.cpp
  timeStepperSSBDF3.cpp
)

add_library(timeStepper STATIC ${timeStepperlib-src})
target_include_directories(timeStepper
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(timeStepper PUBLIC mesh OCCA::libocca MPI::MPI_CXX)
set_target_properties(timeStepper PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/timeStepper.hpp)

target_compile_definitions(timeStepper PUBLIC "TIMESTEPPER_DIR=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS timeStepper)
install(DIRECTORY okl/ DESTINATION okl)

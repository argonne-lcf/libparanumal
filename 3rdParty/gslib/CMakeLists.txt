project(gslib)

set(gslib-src
  src/gs.c 
  src/sort.c 
  src/sarray_transfer.c 
  src/sarray_sort.c 
  src/gs_local.c 
  src/fail.c 
  src/crystal.c
  src/comm.c 
  src/tensor.c
  src/fcrystal.c
  src/findpts_local.c
  src/obbox.c
  src/poly.c
  src/lob_bnd.c
  src/findpts_el_3.c
  src/findpts_el_2.c
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMPI")

add_library(gs STATIC ${gslib-src})
target_include_directories(gs 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(gs PUBLIC MPI::MPI_C)
set_target_properties(gs PROPERTIES PUBLIC_HEADER src/gslib.h)

target_compile_definitions(gs PUBLIC "UNDERSCORE" "GLOBAL_LONG_LONG")
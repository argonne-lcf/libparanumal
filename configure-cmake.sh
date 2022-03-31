# !/bin/bash
CC=icx
CXX=icpx

# For now assume we have an external OCCA build 
EXTERNAL_OCCA=ON

# A semicolon separated list of non-default paths to dependencies
PREFIX_PATHS="${CMPROOT}/linux;${CMPROOT}/linux/include/sycl;${CMPROOT}/linux/compiler"

# Default build parameters
: ${BUILD_DIR:=`pwd`/build}
: ${INSTALL_PATH:=`pwd`/install}
: ${BUILD_TYPE:="RelWithDebInfo"}

: ${CC:="gcc"}
: ${CXX:="g++"}

: ${MPICC:="mpicc"}
: ${MPICXX:="mpicxx"}

cmake -S . -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -DCMAKE_PREFIX_PATH=${PREFIX_PATHS} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_C_FLAGS="${CFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DMPI_C_COMPILER=${MPICC} \
  -DMPI_CXX_COMPILER=${MPICXX} \
  -DEXTERNAL_OCCA=${EXTERNAL_OCCA}


#cmake --build ${BUILD_DIR} --parallel 8
#cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}
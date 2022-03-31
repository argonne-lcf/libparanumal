project(meshlib)

set(meshlib-src
  mesh.cpp
  meshBasis1D.cpp
  meshBasisHex3D.cpp
  meshBasisQuad2D.cpp
  meshBasisTet3D.cpp
  meshBasisTri2D.cpp
  meshConnect.cpp
  meshConnectBoundary.cpp
  meshConnectFaceNodes2D.cpp
  meshConnectFaceNodes3D.cpp
  meshCubatureNodesHex3D.cpp
  meshCubatureNodesQuad2D.cpp
  meshCubatureNodesQuad3D.cpp
  meshCubatureNodesTet3D.cpp
  meshCubatureNodesTri2D.cpp
  meshCubatureNodesTri3D.cpp
  meshCubatureSetupHex3D.cpp
  meshCubatureSetupQuad2D.cpp
  meshCubatureSetupTet3D.cpp
  meshCubatureSetupTri2D.cpp
  meshGeometricFactorsHex3D.cpp
  meshGeometricFactorsQuad2D.cpp
  meshGeometricFactorsQuad3D.cpp
  meshGeometricFactorsTet3D.cpp
  meshGeometricFactorsTri2D.cpp
  meshGeometricFactorsTri3D.cpp
  meshGeometricPartition2D.cpp
  meshGeometricPartition3D.cpp
  meshHaloRingSetup.cpp
  meshHaloSetup.cpp
  meshHaloTraceSetup.cpp
  meshMassMatrixApply.cpp
  meshMinCharacteristicLength.cpp
  meshMultiRateHaloTraceSetup.cpp
  meshMultiRateSetup.cpp
  meshOccaSetup.cpp
  meshOccaSetup2D.cpp
  meshOccaSetup3D.cpp
  meshOccaSetupHex3D.cpp
  meshOccaSetupQuad2D.cpp
  meshOccaSetupQuad3D.cpp
  meshOccaSetupTet3D.cpp
  meshOccaSetupTri2D.cpp
  meshOccaSetupTri3D.cpp
  meshParallelConnectNodes.cpp
  meshParallelConnectOpt.cpp
  meshParallelGatherScatterSetup.cpp
  meshParallelReaderHex3D.cpp
  meshParallelReaderQuad2D.cpp
  meshParallelReaderQuad3D.cpp
  meshParallelReaderTet3D.cpp
  meshParallelReaderTri2D.cpp
  meshParallelReaderTri3D.cpp
  meshPartitionStatistics.cpp
  meshPhysicalNodesHex3D.cpp
  meshPhysicalNodesQuad2D.cpp
  meshPhysicalNodesQuad3D.cpp
  meshPhysicalNodesTet3D.cpp
  meshPhysicalNodesTri2D.cpp
  meshPhysicalNodesTri3D.cpp
  meshPlotInterpHex3D.cpp
  meshPlotInterpQuad2D.cpp
  meshPlotInterpQuad3D.cpp
  meshPlotInterpTet3D.cpp
  meshPlotInterpTri2D.cpp
  meshPlotInterpTri3D.cpp
  meshPmlSetup.cpp
  meshReferenceNodesHex3D.cpp
  meshReferenceNodesQuad2D.cpp
  meshReferenceNodesTet3D.cpp
  meshReferenceNodesTri2D.cpp
  meshSettings.cpp
  meshSetup.cpp
  meshSetupBoxHex3D.cpp
  meshSetupBoxQuad2D.cpp
  meshSetupBoxTet3D.cpp
  meshSetupBoxTri2D.cpp
  meshSetupNewDegree.cpp
  meshSetupPmlBoxHex3D.cpp
  meshSetupPmlBoxQuad2D.cpp
  meshSetupPmlBoxTet3D.cpp
  meshSetupPmlBoxTri2D.cpp
  meshSetupRingPatch.cpp
  meshSetupSEMFEM.cpp
  meshSurfaceGeometricFactorsHex3D.cpp
  meshSurfaceGeometricFactorsQuad2D.cpp
  meshSurfaceGeometricFactorsQuad3D.cpp
  meshSurfaceGeometricFactorsTet3D.cpp
  meshSurfaceGeometricFactorsTri2D.cpp
  meshSurfaceGeometricFactorsTri3D.cpp
)

add_library(mesh STATIC ${meshlib-src})
target_include_directories(mesh
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_link_libraries(mesh PUBLIC linAlg ogs OCCA::libocca MPI::MPI_CXX)
set_target_properties(mesh PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/mesh.hpp)

target_compile_definitions(mesh PUBLIC "MESH_DIR=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS mesh)

install(DIRECTORY okl/ DESTINATION okl)

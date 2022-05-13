/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "cns.hpp"

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  if(argc!=2)
    LIBP_ABORT(string("Usage: ./cnsBench setupfile"));

  //create default settings
  platformSettings_t platformSettings(comm);
  meshSettings_t meshSettings(comm);
  cnsSettings_t cnsSettings(comm);

  // set up platform
  platformSettings.changeSetting("THREAD MODEL","dpcpp");
  platform_t platform(platformSettings);

  // set up mesh
  meshSettings.changeSetting("MESH DIMENSION","3");	// 3D elements
  meshSettings.changeSetting("ELEMENT TYPE","6");	// Hex elements
  meshSettings.changeSetting("BOX BOUNDARY FLAG","-1");	// Periodic
  meshSettings.changeSetting("POLYNOMIAL DEGREE","4");
  mesh_t& mesh = mesh_t::Setup(platform, meshSettings, comm);
  

  // Setup cns solver
  cns_t cns(platform,mesh,cnsSettings);

  //Specify solver settings -> cnsSetup
  cns.mu = 0.001;
  cns.gamma = 1.4;
  cns.cubature = 0;	// 1 = Cubature, 0 = Others (Collocation)
  cns.isothermal = 0;  // 1 = True, 0 = False

  cns.Nfields = 4;	// 4 = 3D, 3 = 2D
  cns.Ngrads = 3*3;

  if (cns.cubature) {
    mesh.CubatureSetup();
    mesh.CubatureNodes();
  }

  if(!cns.isothermal) cns.Nfields++;
  
  // From CNS Setup
  dlong NlocalFields = mesh.Nelements*mesh.Np*cns.Nfields;
  dlong NhaloFields  = mesh.totalHaloPairs*mesh.Np*cns.Nfields;
  dlong NlocalGrads = mesh.Nelements*mesh.Np*cns.Ngrads;
  dlong NhaloGrads  = mesh.totalHaloPairs*mesh.Np*cns.Ngrads;

  cns.q = (dfloat*) calloc(NlocalFields+NhaloFields, sizeof(dfloat));
  cns.o_q = platform.malloc((NlocalFields+NhaloFields)*sizeof(dfloat),cns.q);

  cns.gradq = (dfloat*) calloc(NlocalGrads+NhaloGrads, sizeof(dfloat));
  cns.o_gradq = platform.malloc((NlocalGrads+NhaloGrads)*sizeof(dfloat),cns.gradq);

  occa::properties kernelInfo = mesh.props;
  string dataFileName = "cnsGaussian3D.h";
  // sprintf(dataFileName, "cnsGaussian3D.h");
  kernelInfo["includes"] += dataFileName;
  kernelInfo["defines/" "p_Nfields"] = cns.Nfields;
  kernelInfo["defines/" "p_Ngrads"]  = cns.Ngrads;

  // Work-block parameters
  int blockMax = 512;
  int NblockV = mymax(1, blockMax/mesh.Np);
  kernelInfo["defines/" "p_NblockV"]= NblockV;

  int maxNodes = mymax(mesh.Np, (mesh.Nfp*mesh.Nfaces));
  int NblockS = mymax(1, blockMax/maxNodes);
  kernelInfo["defines/" "p_NblockS"]= NblockS;

  // run
  occa::memory o_rhsq;
  o_rhsq = platform.malloc(NlocalFields*sizeof(dfloat));

  char fileName[BUFSIZ], kernelName[BUFSIZ];
  sprintf(fileName, DCNS "/okl/cnsSurfaceHex3D.okl");
  sprintf(kernelName, "cnsSurfaceHex3D");
  cns.surfaceKernel = platform.buildKernel(fileName, kernelName, kernelInfo);

  dfloat time = 0.0;
  cns.surfaceKernel(mesh.Nelements,
		    mesh.o_sgeo,
		    mesh.o_LIFT,
		    mesh.o_vmapM,
		    mesh.o_vmapP,
		    mesh.o_EToB,
		    mesh.o_x,
		    mesh.o_y,
		    mesh.o_z,
		    time,
		    cns.mu,
		    cns.gamma,
		    cns.o_q,
		    cns.o_gradq,
		    o_rhsq);

  printf(" Run successful ... ");
  // close down MPI
  MPI_Finalize();
  return LIBP_SUCCESS;
}

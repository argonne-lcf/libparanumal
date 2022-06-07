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
#include "bench.hpp"

#include<iostream>
#include<chrono>
#include<random>

int main(int argc, char **argv)
{
  const char* benchmark_kernel[20] = {"SurfaceHex3D","VolumeHex3D", "GradSurfaceHex3D"};

  // start up MPI
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  if(argc<4)
    LIBP_ABORT(string("Usage: ./cnsBench kernel Ngrids Norder kernel_language"));

  int iopt = std::atoi(argv[1]);   // Choose kernel to run
  std::string NX(argv[2]);         // Number of elements in each direction
  std::string Norder(argv[3]);     // Polynomial order
  std::string kernel_lang = "okl";
  if(argc>=5)
    kernel_lang = argv[4];
    //std::string kernel_lang(argv[4]); // kernel language - okl/native
  std::size_t tile_dim = 16;
  if(argc>=6)
    tile_dim = std::stoul(argv[5]);
  

  //create default settings
  platformSettings_t platformSettings(comm);
  meshSettings_t meshSettings(comm);
  cnsSettings_t cnsSettings(comm);

  // set up platform
  platformSettings.changeSetting("THREAD MODEL","dpcpp");
  platform_t platform(platformSettings);

  // set up mesh
  meshSettings.changeSetting("MESH DIMENSION","3");	// 3D elements
  meshSettings.changeSetting("ELEMENT TYPE","12");	// Hex elements
  meshSettings.changeSetting("BOX BOUNDARY FLAG","-1");	// Periodic
  meshSettings.changeSetting("POLYNOMIAL DEGREE",Norder);
  meshSettings.changeSetting("BOX NX",NX);
  meshSettings.changeSetting("BOX NY",NX);
  meshSettings.changeSetting("BOX NZ",NX);
  mesh_t& mesh = mesh_t::Setup(platform, meshSettings, comm);
  //meshSettings.report();
  
  // Setup cns solver
  cns_t cns(platform,mesh,cnsSettings);

  //Specify solver settings -> cnsSetup
  cns.mu = 0.01;
  cns.gamma = 1.4;
  cns.cubature = 0;	// 1 = Cubature, 0 = Others (Collocation)
  cns.isothermal = 0;   // 1 = True, 0 = False

  cns.Nfields = 4;	// 4 = 3D, 3 = 2D
  cns.Ngrads = 3*3;

  if (cns.cubature) {
    mesh.CubatureSetup();
    mesh.CubatureNodes();
  }

  if(!cns.isothermal) cns.Nfields++;
  
  // From CNS Setup
  int Nelements = mesh.Nelements; 
  int Np = mesh.Np;
  int Nfields = cns.Nfields;
  dlong NlocalFields = mesh.Nelements*mesh.Np*cns.Nfields;
  dlong NlocalGrads  = mesh.Nelements*mesh.Np*cns.Ngrads;
  dlong NhaloFields  = mesh.totalHaloPairs*mesh.Np*cns.Nfields;
  dlong NhaloGrads   = mesh.totalHaloPairs*mesh.Np*cns.Ngrads;

  // Initialize required arrays
  std::random_device rd;   // Used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister engine seeded with rd
  std::uniform_real_distribution<dfloat> distribution(-1.0,1.0);

  cns.q = (dfloat*) calloc(NlocalFields+NhaloFields, sizeof(dfloat));
  cns.gradq = (dfloat*) calloc(NlocalGrads+NhaloGrads, sizeof(dfloat));

  for(int e=0;e<Nelements;++e) {
    for(int n=0;n<Np;++n) {
      dlong id = e*Np*Nfields + n;
      cns.q[id+0*Np] = 1.0 + distribution(gen); // rho
      cns.q[id+1*Np] = distribution(gen);       // rho*u
      cns.q[id+2*Np] = distribution(gen);       // rho*v
      cns.q[id+3*Np] = distribution(gen);       // rho*w
      cns.q[id+4*Np] = 1.0 + distribution(gen); // rho*etotal
    }
  }

  cns.o_q = platform.malloc((NlocalFields+NhaloFields)*sizeof(dfloat),cns.q);
  cns.o_gradq = platform.malloc((NlocalGrads+NhaloGrads)*sizeof(dfloat),cns.gradq);

  occa::properties kernelInfo = mesh.props;
  string dataFileName = "cnsGaussian3D.h";
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

  // Kernel setup
  occa::memory o_rhsq;
  o_rhsq = platform.malloc(NlocalFields*sizeof(dfloat));
  occa::kernel baseline_kernel;

  char fileName[BUFSIZ], kernelName[BUFSIZ];
  if(iopt==1)
  {
    sprintf(kernelName, "cnsSurfaceHex3D");
    if (kernel_lang=="okl") {
      sprintf(fileName, DCNS "/okl/cnsSurfaceHex3D.okl");
    }
    else if (kernel_lang=="native") {
      sprintf(fileName, DCNS "/okl/cnsSurfaceHex3D.cpp");
      // occa::json kernel_properties{platform.devicekernelProperties()};
      kernelInfo["okl/enabled"] = false;
    }
    cns.surfaceKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
    int Nq = mesh.N + 1;
    occa::dim innerDims(Nq,Nq,tile_dim);
    occa::dim outerDims(Nelements/tile_dim);
    cns.surfaceKernel.setRunDims(outerDims,innerDims);
  }
  else if(iopt==2)
  {
    sprintf(kernelName, "cnsVolumeHex3D");
    if (kernel_lang=="okl") {
      sprintf(fileName, DCNS "/okl/cnsVolumeHex3D.okl");
    }
    else if (kernel_lang=="native") {
      sprintf(fileName, DCNS "/okl/cnsVolumeHex3D.cpp");
      kernelInfo["okl/enabled"] = false;
    }
    cns.volumeKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
    // int Nq = mesh.N + 1;
    // occa::dim innerDims(Nq,Nq,tile_dim);
    // occa::dim outerDims(Nelements/tile_dim);
    // cns.volumeKernel.setRunDims(outerDims,innerDims);
  }
  else
  {
    sprintf(kernelName, "cnsGradSurfaceHex3D");
    if (kernel_lang=="okl") {
      sprintf(fileName, DCNS "/okl/cnsGradSurfaceHex3D.okl");
    }
    else if (kernel_lang=="native") {
      sprintf(fileName, DCNS "/okl/cnsGradSurfaceHex3D.cpp");
      kernelInfo["okl/enabled"] = false;      
    }
    cns.gradSurfaceKernel = platform.buildKernel(fileName, kernelName, kernelInfo);

    sprintf(kernelName, "cnsGradSurfaceHex3D");
    sprintf(fileName, DCNS "/okl/cnsGradSurfaceHex3D.okl");
    baseline_kernel = platform.buildKernel(fileName,kernelName,kernelInfo);

    int Nq = mesh.N + 1;
    occa::dim innerDims(Nq,Nq,tile_dim);
    occa::dim outerDims(Nelements/tile_dim);
    cns.gradSurfaceKernel.setRunDims(outerDims,innerDims);
  }

  dfloat simulation_time = 0.0;

  // Run kernel and measure runtimes
  int ntrials = 1000;//std::atoi(argv[2]);
  std::vector<dfloat> walltimes(ntrials);
  if(iopt==1) {
    for(size_t trial{}; trial < 10; ++trial) {
      cns.surfaceKernel(mesh.Nelements,
		        mesh.o_sgeo,
			mesh.o_LIFT,
			mesh.o_vmapM,
			mesh.o_vmapP,
			mesh.o_EToB,
			mesh.o_x,
			mesh.o_y,
			mesh.o_z,
			simulation_time,
			cns.mu,
			cns.gamma,
			cns.o_q,
			cns.o_gradq,
			o_rhsq);
      platform.device.finish();
    }

    for(size_t trial{}; trial < ntrials; ++trial) {
      auto start_time = std::chrono::high_resolution_clock::now();

      cns.surfaceKernel(mesh.Nelements,
          	      mesh.o_sgeo,
          	      mesh.o_LIFT,
          	      mesh.o_vmapM,
          	      mesh.o_vmapP,
          	      mesh.o_EToB,
          	      mesh.o_x,
          	      mesh.o_y,
          	      mesh.o_z,
          	      simulation_time,
          	      cns.mu,
          	      cns.gamma,
          	      cns.o_q,
          	      cns.o_gradq,
          	      o_rhsq);
      platform.device.finish();

      auto finish_time = std::chrono::high_resolution_clock::now();
      walltimes[trial] = std::chrono::duration<double,std::milli>(finish_time-start_time).count();
    }
  }

  else if(iopt==2) {
    for(size_t trial{}; trial < 10; ++trial) {
      cns.volumeKernel (mesh.Nelements,
		        mesh.o_vgeo,
			mesh.o_D,
			mesh.o_x,
			mesh.o_y,
			mesh.o_z,
			simulation_time,
			cns.mu,
			cns.gamma,
			cns.o_q,
			cns.o_gradq,
			o_rhsq);
      platform.device.finish();
    }

    for(size_t trial{}; trial < ntrials; ++trial) {
      auto start_time = std::chrono::high_resolution_clock::now();

      cns.volumeKernel (mesh.Nelements,
                        mesh.o_vgeo,
                        mesh.o_D,
                        mesh.o_x,
                        mesh.o_y,
                        mesh.o_z,
                        simulation_time,
                        cns.mu,
                        cns.gamma,
                        cns.o_q,
                        cns.o_gradq,
                        o_rhsq);
      platform.device.finish();

      auto finish_time = std::chrono::high_resolution_clock::now();
      walltimes[trial] = std::chrono::duration<dfloat,std::milli>(finish_time-start_time).count();
    }      
  }

  else {
    // Obtaining result from baseline kernel
    dfloat *baseline_result  = (dfloat*) calloc(NlocalGrads+NhaloGrads,sizeof(dfloat));
    dfloat *optimized_result = (dfloat*) calloc(NlocalGrads+NhaloGrads,sizeof(dfloat));
    baseline_kernel(mesh.Nelements,mesh.o_sgeo,mesh.o_LIFT,mesh.o_vmapM,mesh.o_vmapP,mesh.o_EToB,mesh.o_x,mesh.o_y,mesh.o_z,
		    simulation_time,cns.mu,cns.gamma,cns.o_q,cns.o_gradq);
    cns.o_gradq.copyTo(baseline_result);

    for(size_t trial{}; trial < 10; ++trial) {
      cns.o_gradq.copyFrom(optimized_result);  // Resetting device memory to zero
      cns.gradSurfaceKernel(mesh.Nelements,
                            mesh.o_sgeo,
                            mesh.o_LIFT,
                            mesh.o_vmapM,
                            mesh.o_vmapP,
                            mesh.o_EToB,
                            mesh.o_x,
                            mesh.o_y,
                            mesh.o_z,
                            simulation_time,
                            cns.mu,
                            cns.gamma,
                            cns.o_q,
                            cns.o_gradq);
      platform.device.finish();
    }
    cns.o_gradq.copyTo(optimized_result);
    if(benchmark::validate(baseline_result,optimized_result,NlocalGrads+NhaloGrads))
       std::cout<<" Validation check passed...\n";

    for(size_t trial{}; trial < ntrials; ++trial) {
      auto start_time = std::chrono::high_resolution_clock::now();
      cns.gradSurfaceKernel(mesh.Nelements,
                            mesh.o_sgeo,
                            mesh.o_LIFT,
                            mesh.o_vmapM,
                            mesh.o_vmapP,
                            mesh.o_EToB,
                            mesh.o_x,
                            mesh.o_y,
                            mesh.o_z,
                            simulation_time,
                            cns.mu,
                            cns.gamma,
                            cns.o_q,
                            cns.o_gradq);
      platform.device.finish();

      auto finish_time = std::chrono::high_resolution_clock::now();
      walltimes[trial] = std::chrono::duration<dfloat,std::milli>(finish_time-start_time).count();
    }
  }
  
  auto walltime_stats = benchmark::calculateStatistics(walltimes);

  // Print results
  std::cout<<" BENCHMARK:\n";
  std::cout<<" - Kernel Name : "<<std::string(benchmark_kernel[iopt-1])<<std::endl;
  std::cout<<" - Backend API : "<<platform.device.mode()<<std::endl;
  std::cout<<" PARAMETERS :\n";
  std::cout<<" - Number of elements : "<<mesh.Nelements<<std::endl;
  std::cout<<" - Polynomial degree  : "<<mesh.N<<std::endl;
  std::cout<<" - Number of trials   : "<<ntrials<<std::endl;
  std::cout<<" RUNTIME STATISTICS:\n";
  std::cout<<" - Mean : "<<std::scientific<<walltime_stats.mean   <<" ms\n";
  std::cout<<" - Min  : "<<std::scientific<<walltime_stats.min    <<" ms\n";
  std::cout<<" - Max  : "<<std::scientific<<walltime_stats.max    <<" ms\n";
  std::cout<<" - Stdv : "<<std::scientific<<walltime_stats.stddev <<" ms\n";
  std::cout<<std::endl;

  // close down MPI
  MPI_Finalize();
  return LIBP_SUCCESS;

}

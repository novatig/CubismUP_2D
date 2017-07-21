//
//  Sim_FSI_Gravity.cpp
//  CubismUP_2D
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "Sim_FSI_Gravity.h"

#include "ProcessOperatorsOMP.h"
#include "CoordinatorIC.h"
#include "CoordinatorAdvection.h"
#include "CoordinatorDiffusion.h"
#include "CoordinatorShape.h"
#include "CoordinatorPressure.h"
#include "CoordinatorGravity.h"

void Sim_FSI_Gravity::_diagnostics()
{
  vector<BlockInfo> vInfo = grid->getBlocksInfo();
  shape->_diagnostics(uBody[0],uBody[1],omegaBody,vInfo,nu,time,step,lambda);
}

void Sim_FSI_Gravity::_ic()
{
  // setup initial conditions
  CoordinatorIC coordIC(shape, uinfx, uinfy, grid);
  profiler.push_start(coordIC.getName());
  coordIC(0);

  vector<BlockInfo> vInfo = grid->getBlocksInfo();
  #pragma omp parallel for schedule(static)
  for(int i=0; i<vInfo.size(); i++) {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      b(ix,iy).rho = 1;
      b(ix,iy).tmp = 0;
    }
  }

  shape->create(vInfo);

  stringstream ss;
  ss << path2file << "-IC.vti";
  dumper.Write(*grid, ss.str());
  profiler.pop_stop();
}

double Sim_FSI_Gravity::_nonDimensionalTime()
{
  return time; // how to nondimensionalize here? based on Galileo number?
}

#ifdef RL_MPI_CLIENT
Sim_FSI_Gravity::Sim_FSI_Gravity(Communicator*const comm,const int argc, const char ** argv) :
communicator(comm),
#else
Sim_FSI_Gravity::Sim_FSI_Gravity(const int argc, const char ** argv) :
#endif
Simulation_FSI(argc, argv)
{
  cout << "=================================================================\n";
  cout << "\t\tFlow past a falling obstacle\n";
  cout << "=================================================================\n";
}

Sim_FSI_Gravity::~Sim_FSI_Gravity() { }

void Sim_FSI_Gravity::init()
{
  Simulation_FSI::init();

  // simulation settings
  nu = parser("-nu").asDouble(1e-2);
  minRho = min((Real)1.,shape->getMinRhoS());

  gravity[1] = -parser("-g").asDouble(9.8);
  uinfx = parser("-uinfx").asDouble(0);
  uinfy = parser("-uinfy").asDouble(0);

  const Real aspectRatio = (Real)bpdx/(Real)bpdy;
  Real center[2] = {
      parser("-xpos").asDouble(.5*aspectRatio),
      parser("-ypos").asDouble(.85)
  };
  shape->setCentroid(center);
  shape->time_ptr = &time;
  shape->grid_ptr = grid;
  #ifdef RL_MPI_CLIENT
  shape->communicator = communicator;
  #endif

  _ic();


  pipeline.clear();

  pipeline.push_back(
    new CoordinatorComputeShape(&uBody[0], &uBody[1], &omegaBody, shape, grid)
  );

  #ifndef _MULTIPHASE_
  pipeline.push_back(
    new CoordinatorAdvection<Lab>(&uBody[0], &uBody[1], grid)
  );
  #else
  pipeline.push_back(
    new CoordinatorAdvection<Lab>(&uBody[0], &uBody[1], grid, 1)
  );
  #endif

  pipeline.push_back(
    new CoordinatorDiffusion<Lab>(nu, &uBody[0], &uBody[1], &dragV, grid)
  );

  pipeline.push_back(
    new CoordinatorGravity(gravity, grid));

  pipeline.push_back(
    new CoordinatorBodyVelocities(&uBody[0], &uBody[1], &omegaBody, shape, &lambda, grid)
  );

  pipeline.push_back(
    new CoordinatorPenalization(&uBody[0], &uBody[1], &omegaBody, shape, &lambda, grid)
  );

  pipeline.push_back(
    new CoordinatorPressure<Lab>(minRho, gravity, &uBody[0], &uBody[1], &dragP[0], &dragP[1], &step, grid)
  );

  pipeline.push_back(
    new CoordinatorComputeForces(&uBody[0], &uBody[1], &omegaBody, shape, &time, &nu, &step, &bDump, grid)
  );

  //#ifdef _MOVING_FRAME_
  pipeline.push_back(
    new CoordinatorFadeOut(&uBody[0], &uBody[1], uinfx, uinfy, grid)
  );
  //#endif

  cout << "Coordinator/Operator ordering:\n";
  for (int c=0; c<pipeline.size(); c++)
    cout << "\t" << pipeline[c]->getName() << endl;
}

void Sim_FSI_Gravity::simulate()
{
  const int sizeX = bpdx * FluidBlock::sizeX;
  const int sizeY = bpdy * FluidBlock::sizeY;

  double uOld = 0, vOld = 0;
  double nextDumpTime = time;
  double maxU = uBody[0];
  double maxA = 0;

  while (true)
  {
    vector<BlockInfo> vInfo = grid->getBlocksInfo();

    // choose dt (CFL, Fourier)
    profiler.push_start("DT");
    maxU = findMaxUOMP(vInfo,*grid);

    const Real maxMu = nu / min(shape->getMinRhoS(),(Real)1);
    const Real maxUbody = max(abs(uBody[0]),abs(uBody[1]));

    dtFourier = CFL*vInfo[0].h_gridpoint*vInfo[0].h_gridpoint / maxMu;
    dtCFL  = maxU < 2.2e-16 ? 1 : CFL*vInfo[0].h_gridpoint/abs(maxU);
    dt = min(dtCFL,dtFourier);

    #ifndef _MOVING_FRAME_
      dtBody = maxUbody < 2.2e-16 ? 1 : CFL*vInfo[0].h_gridpoint/maxUbody;
      dt = min(min(dtCFL,dtFourier),dtBody);
    #endif

    assert(!std::isnan(maxU));
    assert(!std::isnan(maxA));
    assert(!std::isnan(uBody[0]));
    assert(!std::isnan(uBody[1]));

    lambda = 10./dt;

    cout << "time, dt (Fourier, CFL, body): "
      <<time<<" "<<dt<<" "<<dtFourier<<" "<<dtCFL<<" "<<dtBody<<endl;
    profiler.pop_stop();


    assert(dt>2.2e-16);
    bDump = dumpTime>0. && time+dt>nextDumpTime;

    for (int c=0; c<pipeline.size(); c++)
    {
      profiler.push_start(pipeline[c]->getName());
      (*pipeline[c])(dt);
      profiler.pop_stop();
      stringstream ss;
      ss << path2file << "avemaria_" << pipeline[c]->getName() << "_";
      ss << std::setfill('0') << std::setw(7) << step;
      ss  << ".vti";
      //      cout << ss.str() << endl;
      //      _dump(ss);
    }

    time += dt;
    step++;

    if(0) {
      // this test only works for constant density disks as it is written now
      const double accMy = (uBody[1]-vOld)/dt;
      const double accMx = (uBody[0]-uOld)/dt;
      vOld = uBody[1];
      uOld = uBody[0];
      const double accT = (shape->getMinRhoS()-1)/(shape->getMinRhoS()+1)*gravity[1];
      const double accN = (shape->getMinRhoS()-1)/(shape->getMinRhoS()  )*gravity[1];
      //if (verbose)
      cout<<"Acceleration with added mass (measured x,y, expected, no added mass)\t"
          <<accMx<<"\t"<<accMy<<"\t"<<accT<<"\t"<<accN<<endl;
      stringstream ss;
      ss<<path2file<<"_addedmass.dat";
      ofstream myfile(ss.str(), fstream::app);
      myfile<<step<<" "<<time<<" "<<accMx<<" "<<accMy<<" "<<accT<<" "<<accN<<endl;
    }

    // compute diagnostics
    if (step % 10 == 0)
    {
      profiler.push_start("Diagnostics");
      _diagnostics();
      profiler.pop_stop();
    }

    //dump some time steps every now and then
    profiler.push_start("Dump");
    if(bDump) shape->characteristic_function(grid->getBlocksInfo());
    _dump(nextDumpTime);
    profiler.pop_stop();

    if (step % 100 == 0)
      profiler.printSummary();

    // check nondimensional time
    if ((endTime>0 && _nonDimensionalTime() > endTime) || (nsteps!=0 && step>=nsteps))
    {
      profiler.printSummary();
      exit(0);
    }
  }
}

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
#include "CoordinatorMultistep.h"
#include "CoordinatorDiffusion.h"
#include "CoordinatorShape.h"
#include "CoordinatorPressure.h"
#include "CoordinatorGravity.h"

void Sim_FSI_Gravity::_diagnostics()
{
  //shape->_diagnostics(sim);
}

Sim_FSI_Gravity::Sim_FSI_Gravity(int argc, char ** argv) :
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

  // setup initial conditions
  CoordinatorIC coordIC(sim);
  profiler.push_start(coordIC.getName());
  coordIC(0);
  profiler.pop_stop();

  pipeline.clear();

  pipeline.push_back( new CoordinatorComputeShape(sim) );

  pipeline.push_back( new CoordinatorVelocities(sim) );

  pipeline.push_back( new CoordinatorPenalization(sim) );

  #if 1 // in one sweep advect, diffuse, add hydrostatic
    pipeline.push_back( new CoordinatorMultistep<Lab>(sim) );
  #else
    pipeline.push_back( new CoordinatorAdvection<Lab>(sim) );

    pipeline.push_back( new CoordinatorDiffusion<Lab>(sim) );

    pipeline.push_back( new CoordinatorGravity(sim) );
  #endif

  pipeline.push_back( new CoordinatorPressure<Lab>(sim) );

  pipeline.push_back( new CoordinatorComputeForces(sim) );

  #ifndef FREESPACE
    pipeline.push_back( new CoordinatorFadeOut(sim) );
  #endif

  cout << "Coordinator/Operator ordering:\n";
  for (size_t c=0; c<pipeline.size(); c++)
    cout << "\t" << pipeline[c]->getName() << endl;
}

double Sim_FSI_Gravity::calcMaxTimestep()
{
  const Real maxU = findMaxUOMP( sim ); assert(maxU>=0);
  const double h = sim.getH();
  const double dtFourier = h*h/sim.nu, dtCFL = maxU<2.2e-16? 1 : h/maxU;
  sim.dt = sim.CFL * std::min(dtCFL, dtFourier);

  const double maxUb = sim.maxRelSpeed();
  const double dtBody = maxUb<2.2e-16? 1 : h/maxUb;
  sim.dt = std::min( sim.dt, sim.CFL*dtBody );

  if(sim.dlm > 1) sim.lambda = sim.dlm / sim.dt;

  if (sim.step < 100)
  {
    const double x = (sim.step+1)/100;
    const double logCFL = std::log(sim.CFL);
    const double rampCFL = std::exp(-5*(1-x) + logCFL * x);
    sim.dt = rampCFL*std::min({dtCFL, dtFourier, dtBody});
  }
  #ifndef RL_TRAIN
  if(sim.verbose)
    cout << "time, dt (Fourier, CFL, body): "
    <<sim.time<<" "<<sim.dt<<" "<<dtFourier<<" "<<dtCFL<<" "<<dtBody<<" "<<sim.uinfx<<" "<<sim.uinfy<<endl;
  #endif

  return sim.dt;
}

bool Sim_FSI_Gravity::advance(const double dt)
{
  assert(dt>2.2e-16);
  const bool bDump = sim.bDump();

  for (size_t c=0; c<pipeline.size(); c++)
  {
    profiler.push_start(pipeline[c]->getName());
    (*pipeline[c])(sim.dt);
    profiler.pop_stop();
    // stringstream ss;
    // ss << path2file << "avemaria_" << pipeline[c]->getName() << "_";
    // ss << std::setfill('0') << std::setw(7) << step << ".vti";
    //      cout << ss.str() << endl;
    //      _dump(ss);
  }

  sim.time += sim.dt;
  sim.step++;

  // compute diagnostics
  #ifndef RL_TRAIN
  if (sim.step % 10 == 0)
  {
    profiler.push_start("Diagnostics");
    _diagnostics();
    profiler.pop_stop();
  }
  #endif

  //dump some time steps every now and then
  profiler.push_start("Dump");
  if(bDump) {
    sim.registerDump();
    _dump();
  }
  profiler.pop_stop();

  if (sim.step % 100 == 0 && sim.verbose) {
    profiler.printSummary();
    profiler.reset();
  }

  const bool bOver = sim.bOver();
  if(bOver) profiler.printSummary();
  return bOver;
}

//
//  Simulation_Fluid.h
//  CubismUP_2D
//
//  Base class for fluid simulations from which any fluid simulation case should inherit
//  Contains the base structure and interface that any fluid simulation class should have
//
//  Created by Christian Conti on 3/25/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "Definitions.h"
//#include "ProcessOperatorsOMP.h"
#include "GenericCoordinator.h"
#include "GenericOperator.h"

#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

class Simulation_Fluid
{
 public:
  SimulationData sim;
 protected:
  ArgumentParser parser;
  Profiler profiler;
  vector<GenericCoordinator*> pipeline;
  #ifdef USE_VTK
    SerializerIO_ImageVTK<FluidGrid, FluidVTKStreamer> dumper;
  #endif
  virtual void _diagnostics() = 0;

  virtual void _dump(string fname = "") {
    stringstream ss;
    ss << sim.path2file << "avemaria_" << fname;
    ss << std::setfill('0') << std::setw(7) << sim.step;
    ss << ".vti";
    cout << ss.str() << endl;
    #ifdef USE_VTK
      dumper.Write( *(sim.grid), ss.str() );
    #else
      DumpHDF5<FluidGrid,StreamerVelocityVector>(*(sim.grid), sim.step, sim.time, ss.str(), sim.path4serialization);
      DumpHDF5<FluidGrid,StreamerPressure>(*(sim.grid), sim.step, sim.time, ss.str(), sim.path4serialization);
      DumpHDF5<FluidGrid,StreamerRho>(*(sim.grid), sim.step, sim.time, ss.str(), sim.path4serialization);
    #endif
  }

  void _serialize()
  {
    /*
    stringstream ss;
    ss << path4serialization << "Serialized-" << bPing << ".dat";
    cout << ss.str() << endl;

    stringstream serializedGrid;
    serializedGrid << "SerializedGrid-" << bPing << ".grid";
    //DumpZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);

    bPing = !bPing;
    */
  }

  void _deserialize()
  {
    /*
    stringstream ss0, ss1, ss;
    struct stat st0, st1;
    ss0 << path4serialization << "Serialized-0.dat";
    ss1 << path4serialization << "Serialized-1.dat";
    stat(ss0.str().c_str(), &st0);
    stat(ss1.str().c_str(), &st1);

    grid = new FluidGrid(bpdx,bpdy,1);
    assert(grid != NULL);
    {
      stringstream serializedGrid;
      serializedGrid << "SerializedGrid-" << bPing << ".grid";
      ReadZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
    }
    */
  }

 public:
  Simulation_Fluid(const int argc, char ** argv) : parser(argc,argv) { }

  virtual ~Simulation_Fluid()
  {
    while( not pipeline.empty() ) {
      GenericCoordinator * g = pipeline.back();
      pipeline.pop_back();
      if(g not_eq nullptr) delete g;
    }
  }

  void reset() {
    sim.resetAll();
  }

  virtual void init()
  {
    sim.bRestart = parser("-restart").asBool(false);
    cout << "bRestart is " << sim.bRestart << endl;

    // initialize grid
    parser.set_strict_mode();
    const int bpdx = parser("-bpdx").asInt();
    const int bpdy = parser("-bpdy").asInt();
    sim.grid = new FluidGrid(bpdx, bpdy, 1);
    assert( sim.grid not_eq nullptr );

    // simulation ending parameters
    parser.unset_strict_mode();
    sim.nsteps = parser("-nsteps").asInt(0);
    sim.endTime = parser("-tend").asDouble(0);

    // output parameters
    sim.dumpFreq = parser("-fdump").asDouble(0);
    sim.dumpTime = parser("-tdump").asDouble(0);

    sim.path2file = parser("-file").asString("./");
    sim.path4serialization = parser("-serialization").asString(sim.path2file);
    sim.bFreeSpace = parser("-bFreeSpace").asInt(1);

    // simulation settings
    sim.CFL = parser("-CFL").asDouble(.1);
    sim.lambda = parser("-lambda").asDouble(1e5);
    sim.dlm = parser("-dlm").asDouble(10.);
    sim.nu = parser("-nu").asDouble(1e-2);

    sim.verbose = parser("-verbose").asInt(1);
    sim.muteAll = parser("-muteAll").asInt(0);//stronger silence, not even files
    if(sim.muteAll) sim.verbose = 0;
  }

  virtual bool advance(const double dt) = 0;
  virtual double calcMaxTimestep() = 0;

  void simulate()
  {
    while (1) {
        profiler.push_start("DT");
        const double dt = calcMaxTimestep();
        profiler.pop_stop();

        if (advance(dt)) break;
    }
  }
};

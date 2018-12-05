//
//  Sim_FSI_Gravity.h
//  CubismUP_2D
//
//  Class for the simulation of gravity driven FSI
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once
#include "SimulationData.h"
#include "Operator.h"

#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

class Profiler;

class Simulation
{
 public:
  SimulationData sim;
 protected:
  ArgumentParser parser;
  Profiler * profiler = nullptr;
  std::vector<Operator*> pipeline;

  void createShapes();
  void parseRuntime();
  void dump(string fname = "");
  // should this stuff be moved? - serialize method will do that
  //void _dumpSettings(ostream& outStream);

public:
  Simulation(int argc, char ** argv);
  ~Simulation();

  void reset();
  void init();
  void simulate();
  double calcMaxTimestep();
  bool advance(const double DT);

  const vector<Shape*>& getShapes() { return sim.shapes; }
};

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "SimulationData.h"
#include "Operator.h"

class Profiler;

class Simulation
{
 public:
  SimulationData sim;
 protected:
  cubism::ArgumentParser parser;
  std::vector<Operator*> pipeline;

  void createShapes();
  void parseRuntime();
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

  const std::vector<Shape*>& getShapes() { return sim.shapes; }
};

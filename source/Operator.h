//
//  GenericCoordinator.h
//  CubismUP_2D
//
//  This class serves as the interface for a coordinator object
//  A coordinator object schedules the processing of blocks with its operator
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "SimulationData.h"

class Operator
{
protected:
  SimulationData& sim;
  const vector<BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  const size_t Nblocks = velInfo.size();

public:
  Operator(SimulationData& s) : sim(s) { }
  virtual ~Operator() {}
  virtual void operator()(const double dt) = 0;

  virtual string getName() = 0;
};

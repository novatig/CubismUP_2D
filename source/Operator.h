//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
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

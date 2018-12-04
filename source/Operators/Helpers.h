//
//  ProcessOperatorsOMP.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "../Operator.h"

class findMaxU
{
  SimulationData& sim;
  const vector<BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  const size_t Nblocks = velInfo.size();
 public:
  findMaxU(SimulationData& s) : sim(s) { }

  Real run() const;

  string getName()
  {
    return "findMaxU";
  }
};

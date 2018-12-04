//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "../Operator.h"

class RKstep2 : public Operator
{
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& pRHSInfo  = sim.pRHS->getBlocksInfo();

 public:
  RKstep2(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "RKstep2";
  }
};

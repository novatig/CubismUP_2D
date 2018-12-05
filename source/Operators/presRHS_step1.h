//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

/*
This operator assumes that obects have put signed distance on the grid
*/
#pragma once

#include "../Operator.h"

class presRHS_step1 : public Operator
{
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();

 public:
  presRHS_step1(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "presRHS_step1";
  }
};

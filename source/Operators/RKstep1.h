//
//  OperatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "../Operator.h"

class RKstep1 : public Operator
{
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& pRHSInfo = sim.pRHS->getBlocksInfo();

 public:
  RKstep1(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "RKstep1";
  }
};

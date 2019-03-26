//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "../Operator.h"

class advDiff_RK : public Operator
{
  const std::vector<cubism::BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& pRHSInfo  = sim.pRHS->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  void step1(const double dt);
  void step2(const double dt);
 public:
  advDiff_RK(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  std::string getName()
  {
    return "advDiff_RK";
  }
};

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../Operator.h"

class Penalization : public Operator
{
  const std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();

 public:
  void operator()(const double dt);

  Penalization(SimulationData& s) : Operator(s) { }

  std::string getName() {
    return "Penalization";
  }
};

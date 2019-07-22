//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "../Operator.h"

class Shape;

class ComputeForces : public Operator
{
  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();

public:
  void operator()(const double dt);

  ComputeForces(SimulationData& s);
  ~ComputeForces() {}

  std::string getName() {
    return "ComputeForces";
  }
};

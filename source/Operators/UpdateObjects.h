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

class UpdateObjects : public Operator
{
  const std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  //const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  //const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  //const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  void integrateMomenta(Shape * const shape) const;
  void penalize(const double dt) const;
  void preventCollidingObstacles() const;
 public:
  UpdateObjects(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  std::string getName()
  {
    return "UpdateObjects";
  }
};

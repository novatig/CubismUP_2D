//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "Fish.h"
class CarlingFish: public Fish
{
 public:
  double getPhase(const double t) const;

  CarlingFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
  void resetAll() override;
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
};

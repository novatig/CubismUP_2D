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
  void integrateForce(Shape * const shape) const;

 public:
  UpdateObjects(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "UpdateObjects";
  }
};

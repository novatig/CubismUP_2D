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

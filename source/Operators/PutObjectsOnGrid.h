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

class PutObjectsOnGrid : public Operator
{
  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  void putChiOnGrid(Shape * const shape) const;
  void putObjectVelOnGrid(Shape * const shape) const;

  // sets pRHS to div(\chi_{t} F_{t}) and tmpV to (\chi_{t} F_{t})
  void presRHS_step1(const double dt) const;

  // computes: - div(\chi_{t+1} Udef_{t+1})
  void presRHS_step2(const double dt) const;

 public:
  PutObjectsOnGrid(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "PutObjectsOnGrid";
  }
};

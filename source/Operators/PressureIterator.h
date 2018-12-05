//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "Operator.h"

class HYPRE_solver;

class PressureIterator : public Operator
{
  const std::vector<BlockInfo>& velInfo   = sim.vel->getBlocksInfo();
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  const std::vector<BlockInfo>& pRHSInfo  = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();

  HYPRE_solver * const pressureSolver;

  Real updatePenalizationForce(const double dt) const;
  void initPenalizationForce(const double dt) const;
  void updatePressureRHS(const double dt) const;

 public:
  void operator()(const double dt);

  PressureIterator(SimulationData& s);
  ~PressureIterator();

  string getName() {
    return "Pressure";
  }
};

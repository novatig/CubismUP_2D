//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../Operator.h"

class PoissonSolver;

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

  PoissonSolver * const pressureSolver;

  Real updatePenalizationForce(const double dt, const int iter) const;
  void initPenalizationForce(const double dt) const;
  void updatePressureRHS(const double dt) const;
  void fadeoutBorder(const double dt) const;

 public:
  void operator()(const double dt);

  PressureIterator(SimulationData& s);
  ~PressureIterator();

  std::string getName() {
    return "PressureIterator";
  }
};

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

class PressureVarRho : public Operator
{
  const Real rho0 = sim.minRho();
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& pOldInfo  = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo  = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();

  PoissonSolver * const pressureSolver;

  void fadeoutBorder(const double dt) const;
  void pressureCorrection(const double dt) const;
  void updatePressureRHS(const double dt) const;

 public:
  void operator()(const double dt);

  PressureVarRho(SimulationData& s);
  ~PressureVarRho();

  std::string getName() {
    return "PressureVarRho";
  }
};

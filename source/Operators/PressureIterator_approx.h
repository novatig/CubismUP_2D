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

class PressureVarRho_approx : public Operator
{
  int oldNsteps = 10000;
  const Real rho0 = sim.minRho();
  PoissonSolver * const pressureSolver;

  void fadeoutBorder(const double dt) const;
  void updatePressureRHS(const double dt) const;
  void finalizePressure(const double dt) const;
  Real penalize(const double dt) const;
  void integrateMomenta(Shape * const shape) const;
  void pressureCorrectionInit(const double dt) const;
  void pressureCorrection(const double dt) const;

 public:
  void operator()(const double dt);

  PressureVarRho_approx(SimulationData& s);
  ~PressureVarRho_approx();

  std::string getName() {
    return "PressureVarRho_approx";
  }
};

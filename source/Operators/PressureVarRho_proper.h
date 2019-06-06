//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../Operator.h"

#ifdef HYPREFFT
class HYPREdirichletVarRho;
#endif
class PoissonSolver;

class PressureVarRho_proper : public Operator
{
  #ifdef HYPREFFT
    HYPREdirichletVarRho * const varRhoSolver;
  #endif
  PoissonSolver * const unifRhoSolver;

  void pressureCorrection(const double dt) const;
  void updatePressureRHS(const double dt) const;

 public:
  void operator()(const double dt);

  PressureVarRho_proper(SimulationData& s);
  ~PressureVarRho_proper();

  std::string getName() {
    return "PressureVarRho_proper";
  }
};

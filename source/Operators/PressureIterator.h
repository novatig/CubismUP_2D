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

class PressureVarRho_iterator : public Operator
{
  #ifdef HYPREFFT
    HYPREdirichletVarRho * const pressureSolver;
  #endif

  void finalizePressure(const double dt) const;
  Real penalize(const double dt) const;
  void integrateMomenta(Shape * const shape) const;
  void pressureCorrection(const double dt) const;
  int oldNsteps = 1000;
 public:
  void operator()(const double dt);

  PressureVarRho_iterator(SimulationData& s);
  ~PressureVarRho_iterator();

  std::string getName() {
    return "PressureVarRho_iterator";
  }
};

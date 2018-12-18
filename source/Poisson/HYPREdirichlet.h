//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "HYPRE_struct_ls.h"
#include "PoissonSolver.h"

class HYPREdirichlet : public PoissonSolver
{
  const bool bPeriodic = false;
  const std::string solver;
  HYPRE_StructGrid     hypre_grid;
  HYPRE_StructStencil  hypre_stencil;
  HYPRE_StructMatrix   hypre_mat;
  HYPRE_StructVector   hypre_rhs;
  HYPRE_StructVector   hypre_sol;
  HYPRE_StructSolver   hypre_solver;
  HYPRE_StructSolver   hypre_precond;
  Real pLast = 0;

 public:
  void solve() override;

  HYPREdirichlet(SimulationData& s);

  string getName() {
    return "hypre";
  }

  ~HYPREdirichlet();
};

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "PoissonSolver.h"
#ifdef HYPREFFT

#include "HYPRE_struct_ls.h"

class HYPREdirichletVarRho : public PoissonSolver
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

 public:
  bool bUpdateMat = true;
  using RowType = Real[5];
  RowType * matAry = new RowType[totNy*totNx];

  void solve(const std::vector<cubism::BlockInfo>& BSRC,
             const std::vector<cubism::BlockInfo>& BDST) override;

  HYPREdirichletVarRho(SimulationData& s);

  std::string getName() {
    return "hypre";
  }

  ~HYPREdirichletVarRho();
};

#endif

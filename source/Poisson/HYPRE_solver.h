//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "../Operator.h"
#include "HYPRE_struct_ls.h"

class HYPRE_solver
{
  SimulationData& sim;
  // total number of DOFs in y and x:
  const size_t totNy = sim.vel->getBlocksPerDimension(1) * _BS_;
  const size_t totNx = sim.vel->getBlocksPerDimension(0) * _BS_;

  // memory buffer for mem transfers to/from hypre:
  Real * const buffer = new Real[totNy * totNx];

  const bool bPeriodic = false;
  const std::string solver = "smg";
  HYPRE_StructGrid     hypre_grid;
  HYPRE_StructStencil  hypre_stencil;
  HYPRE_StructMatrix   hypre_mat;
  HYPRE_StructVector   hypre_rhs;
  HYPRE_StructVector   hypre_sol;
  HYPRE_StructSolver   hypre_solver;
  HYPRE_StructSolver   hypre_precond;
  Real pLast = 0;

  void rhs_cub2lin();

  void sol_lin2cub();

 public:
  void solve();

  HYPRE_solver(SimulationData& s);

  string getName() {
    return "hypre";
  }

  ~HYPRE_solver();
};

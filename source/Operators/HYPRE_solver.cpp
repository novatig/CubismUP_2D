//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "HYPRE_solver.h"

void HYPRE_solver::rhs_cub2lin()
{
  const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const size_t nBlocks = tmpInfo.size();

  #pragma omp parallel for schedule(static, 1)
  for(size_t i=0; i<nBlocks; i++)
  {
    const BlockInfo& info = tmpInfo[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    const size_t blockStart = blocki + totNx * blockj;
    ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      buffer[blockStart + ix + totNx*iy] = b(ix,iy).s;
  }
}

void HYPRE_solver::sol_lin2cub()
{
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const size_t nBlocks = presInfo.size();

  Real _avgP = 0;
  const Real fac = 1.0 / (totNx * totNy);
  #pragma omp parallel for schedule(static, 1) reduction(+:_avgP)
  for(size_t i=0; i<nBlocks; i++)
  {
    const BlockInfo& info = presInfo[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    const size_t blockStart = blocki + totNx * blockj;
    ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      const Real P = buffer[blockStart + ix + totNx*iy];
      b(ix,iy).s = P; _avgP += fac * P;
    }
  }
  {
    for (size_t i = 0; i < totNy*totNx; i++) buffer[i] -= _avgP;
    HYPRE_Int ilower[2] = {0,0};
    HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, buffer);
    pLast = buffer[totNx*totNy-1];
    printf("Avg Pressure:%f\n",_avgP);
  }
}

void HYPRE_solver::solve()
{
  HYPRE_Int ilower[2] = {0,0};
  HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};

  sim.startProfiler("HYPRE_solver_rhs_cub2lin");
  rhs_cub2lin();
  sim.stopProfiler();

  if(not bPeriodic)
  {
    buffer[totNx*totNy-1] = pLast;
    buffer[totNx*(totNy-2) + totNx-1] -= pLast;
    buffer[totNx*(totNy-1) + totNx-2] -= pLast;
  }

  sim.startProfiler("HYPRE_solver_setBoxVals");
  HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, buffer);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_solver_solve");
  if (solver == "gmres")
    HYPRE_StructGMRESSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else if (solver == "smg")
    HYPRE_StructSMGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else
    HYPRE_StructPCGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_solver_getBoxVals");
  HYPRE_StructVectorGetBoxValues(hypre_sol, ilower, iupper, buffer);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_solver_sol_lin2cub");
  sol_lin2cub();
  sim.stopProfiler();
}

HYPRE_solver::HYPRE_solver(SimulationData& s) : sim(s)
{
  HYPRE_Int ilower[2] = {0,0};
  HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};
  const auto COMM = MPI_COMM_SELF;
  // Grid
  HYPRE_StructGridCreate(COMM, 2, &hypre_grid);

  HYPRE_StructGridSetExtents(hypre_grid, ilower, iupper);

  if(bPeriodic)
  {
    // if grid is periodic, this function takes the period
    // length... ie. the grid size.
    HYPRE_StructGridSetPeriodic(hypre_grid, iupper);
  }

  HYPRE_StructGridAssemble(hypre_grid);

  { // Stencil
    HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
    HYPRE_StructStencilCreate(2, 5, &hypre_stencil);
    for (int j = 0; j < 5; ++j)
      HYPRE_StructStencilSetElement(hypre_stencil, j, offsets[j]);
  }

  { // Matrix
    HYPRE_StructMatrixCreate(COMM, hypre_grid, hypre_stencil, &hypre_mat);
    HYPRE_StructMatrixInitialize(hypre_mat);

    // These indices must match to those in the offset array:
    HYPRE_Int inds[5] = {0, 1, 2, 3, 4};
    using RowType = Real[5];
    RowType * vals = new RowType[totNy*totNx];

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < totNy; j++)
    for (size_t i = 0; i < totNx; i++) {
      vals[j*totNx + i][0] = -4; // center
      vals[j*totNx + i][1] =  1; // west
      vals[j*totNx + i][2] =  1; // east
      vals[j*totNx + i][3] =  1; // south
      vals[j*totNx + i][4] =  1; // north
    }

    if(not bPeriodic) // 0 Neumann BC
    {
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < totNx; i++) {
        vals[totNx*(0)       +i][0] += 1; // center
        vals[totNx*(0)       +i][3]  = 0; // south
        vals[totNx*(totNy-1) +i][0] += 1; // center
        vals[totNx*(totNy-1) +i][4]  = 1; // north
      }
      #pragma omp parallel for schedule(static)
      for (size_t j = 0; j < totNy; j++) {
        vals[totNx*j +       0][0] += 1; // center
        vals[totNx*j +       0][1]  = 1; // west
        vals[totNx*j + totNx-1][0] += 1; // center
        vals[totNx*j + totNx-1][2]  = 1; // east
      }

      // set last corner such that last point has pressure pLast
      vals[totNx*(totNy-1) + totNx-1][0] = 1; // center
      vals[totNx*(totNy-1) + totNx-1][1] = 0; // west
      vals[totNx*(totNy-1) + totNx-1][2] = 0; // east
      vals[totNx*(totNy-1) + totNx-1][3] = 0; // south
      vals[totNx*(totNy-1) + totNx-1][4] = 0; // north
      vals[totNx*(totNy-2) + totNx-1][4] = 0;
      vals[totNx*(totNy-1) + totNx-2][2] = 0;
    }

    Real * const linV = (Real*) vals;
    HYPRE_StructMatrixSetBoxValues(hypre_mat, ilower, iupper, 5, inds, linV);
    delete [] vals;
    HYPRE_StructMatrixAssemble(hypre_mat);
  }


  // Rhs and initial guess
  HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_rhs);
  HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_sol);

  HYPRE_StructVectorInitialize(hypre_rhs);
  HYPRE_StructVectorInitialize(hypre_sol);

  {
    memset(buffer, 0, totNx*totNy*sizeof(Real));
    HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, buffer);
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, buffer);
  }

  HYPRE_StructVectorAssemble(hypre_rhs);
  HYPRE_StructVectorAssemble(hypre_sol);

  if (solver == "gmres") {
    printf("Using GMRES solver\n");
    HYPRE_StructGMRESCreate(COMM, &hypre_solver);
    HYPRE_StructGMRESSetTol(hypre_solver, 1e-2);
    HYPRE_StructGMRESSetPrintLevel(hypre_solver, 2);
    HYPRE_StructGMRESSetMaxIter(hypre_solver, 1000);
    HYPRE_StructGMRESSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else if (solver == "smg") {
    printf("Using SMG solver\n");
    HYPRE_StructSMGCreate(COMM, &hypre_solver);
    //HYPRE_StructSMGSetMemoryUse(hypre_solver, 0);
    HYPRE_StructSMGSetMaxIter(hypre_solver, 100);
    HYPRE_StructSMGSetTol(hypre_solver, 1e-2);
    //HYPRE_StructSMGSetRelChange(hypre_solver, 0);
    HYPRE_StructSMGSetPrintLevel(hypre_solver, 3);
    HYPRE_StructSMGSetNumPreRelax(hypre_solver, 1);
    HYPRE_StructSMGSetNumPostRelax(hypre_solver, 1);

    HYPRE_StructSMGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else {
    printf("Using PCG solver\n");
    HYPRE_StructPCGCreate(COMM, &hypre_solver);
    HYPRE_StructPCGSetMaxIter(hypre_solver, 1000);
    HYPRE_StructPCGSetTol(hypre_solver, 1e-2);
    HYPRE_StructPCGSetPrintLevel(hypre_solver, 2);
    if(0)
    { // Use SMG preconditioner
      HYPRE_StructSMGCreate(COMM, &hypre_precond);
      HYPRE_StructSMGSetMaxIter(hypre_precond, 1000);
      HYPRE_StructSMGSetTol(hypre_precond, 0);
      HYPRE_StructSMGSetNumPreRelax(hypre_precond, 1);
      HYPRE_StructSMGSetNumPostRelax(hypre_precond, 1);
      HYPRE_StructPCGSetPrecond(hypre_solver, HYPRE_StructSMGSolve,
                                HYPRE_StructSMGSetup, hypre_precond);
    }
    HYPRE_StructPCGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
}

HYPRE_solver::~HYPRE_solver()
{
  if (solver == "gmres")
    HYPRE_StructGMRESDestroy(hypre_solver);
  else if (solver == "smg")
    HYPRE_StructSMGDestroy(hypre_solver);
  else
    HYPRE_StructPCGDestroy(hypre_solver);
  HYPRE_StructGridDestroy(hypre_grid);
  HYPRE_StructStencilDestroy(hypre_stencil);
  HYPRE_StructMatrixDestroy(hypre_mat);
  HYPRE_StructVectorDestroy(hypre_rhs);
  HYPRE_StructVectorDestroy(hypre_sol);
  delete [] buffer;
}

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "HYPREdirichlet.h"
//#define CONSISTENT

void HYPREdirichlet::solve()
{
  HYPRE_Int ilower[2] = {0,0};
  HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};

  sim.startProfiler("HYPRE_cub2rhs");
  cub2rhs();
  sim.stopProfiler();

  if(not bPeriodic)
  {
    buffer[totNx*totNy-1] = pLast;
    #ifdef CONSISTENT
      static constexpr int SHIFT = 2;
    #else
      static constexpr int SHIFT = 1;
    #endif
    buffer[totNx*(totNy-1-SHIFT) + totNx-1] -= pLast;
    buffer[totNx*(totNy-1) + totNx-1-SHIFT] -= pLast;
  }

  sim.startProfiler("HYPRE_setBoxV");
  HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, buffer);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_solve");
  if (solver == "gmres")
    HYPRE_StructGMRESSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else if (solver == "smg")
    HYPRE_StructSMGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else
    HYPRE_StructPCGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_getBoxV");
  HYPRE_StructVectorGetBoxValues(hypre_sol, ilower, iupper, buffer);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_sol2cub");
  {
    Real avgP = 0;
    const Real fac = 1.0 / (totNx * totNy);
    Real * const nxtGuess = buffer;
    #pragma omp parallel for schedule(static) reduction(+ : avgP)
    for (size_t i = 0; i < totNy*totNx; i++) avgP += fac * nxtGuess[i];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < totNy*totNx; i++) nxtGuess[i] -= avgP;
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, buffer);
    pLast = buffer[totNx*totNy-1];
    printf("Avg Pressure:%f\n", avgP);
  }
  sol2cub();

  sim.stopProfiler();
}

#define STRIDE s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX

HYPREdirichlet::HYPREdirichlet(SimulationData& s) : PoissonSolver(s, STRIDE),
solver("smg")
{
  printf("Employing HYPRE-based Poisson solver with Dirichlet BCs.\n");
  buffer = new Real[totNy * totNx];
  HYPRE_Int ilower[2] = {0,0};
  HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};
  const auto COMM = MPI_COMM_SELF;
  // Grid
  HYPRE_StructGridCreate(COMM, 2, &hypre_grid);

  HYPRE_StructGridSetExtents(hypre_grid, ilower, iupper);

  //HYPRE_Int ghosts[2] = {2, 2};
  //HYPRE_StructGridSetNumGhost(hypre_grid, ghosts);

  if(bPeriodic)
  {
    // if grid is periodic, this function takes the period
    // length... ie. the grid size.
    HYPRE_StructGridSetPeriodic(hypre_grid, iupper);
  }

  HYPRE_StructGridAssemble(hypre_grid);

  { // Stencil
    #ifdef CONSISTENT
      HYPRE_Int offsets[5][2] = {{0,0}, {-2,0}, {2,0}, {0,-2}, {0,2}};
    #else
      HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
    #endif
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
    #ifdef CONSISTENT
      static constexpr Real COEF = 0.25;
    #else
      static constexpr Real COEF = 1;
    #endif

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < totNy; j++)
    for (size_t i = 0; i < totNx; i++) {
      vals[j*totNx + i][0] = -4*COEF; // center
      vals[j*totNx + i][1] =  1*COEF; // west
      vals[j*totNx + i][2] =  1*COEF; // east
      vals[j*totNx + i][3] =  1*COEF; // south
      vals[j*totNx + i][4] =  1*COEF; // north
    }

    if(not bPeriodic) // 0 Neumann BC
    {
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < totNx; i++) {
        // first south row
        vals[totNx*(0)       +i][0] += COEF; // center
        vals[totNx*(0)       +i][3]  = 0; // south
        #ifdef CONSISTENT
        // second south row
        vals[totNx*(1)       +i][0] += COEF; // center
        vals[totNx*(1)       +i][3]  = 0; // south
        #endif

        // first north row
        vals[totNx*(totNy-1) +i][0] += COEF; // center
        vals[totNx*(totNy-1) +i][4]  = 0; // north
        #ifdef CONSISTENT
        // second north row
        vals[totNx*(totNy-2) +i][0] += COEF; // center
        vals[totNx*(totNy-2) +i][4]  = 0; // north
        #endif
      }
      #pragma omp parallel for schedule(static)
      for (size_t j = 0; j < totNy; j++) {
        // first west col
        vals[totNx*j +       0][0] += COEF; // center
        vals[totNx*j +       0][1]  = 0; // west
        #ifdef CONSISTENT
        // second west col
        vals[totNx*j +       1][0] += COEF; // center
        vals[totNx*j +       1][1]  = 0; // west
        #endif
        // first east col
        vals[totNx*j + totNx-1][0] += COEF; // center
        vals[totNx*j + totNx-1][2]  = 0; // east
        #ifdef CONSISTENT
        // second east col
        vals[totNx*j + totNx-2][0] += COEF; // center
        vals[totNx*j + totNx-2][2]  = 0; // east
        #endif
      }

      {
        #ifdef CONSISTENT
          static constexpr int SHIFT = 2;
        #else
          static constexpr int SHIFT = 1;
        #endif
        // set last corner such that last point has pressure pLast
        vals[totNx*(totNy-1) + totNx-1][0] = 1; // center
        vals[totNx*(totNy-1) + totNx-1][1] = 0; // west
        vals[totNx*(totNy-1) + totNx-1][2] = 0; // east
        vals[totNx*(totNy-1) + totNx-1][3] = 0; // south
        vals[totNx*(totNy-1) + totNx-1][4] = 0; // north
        vals[totNx*(totNy-1-SHIFT) + totNx-1][4] = 0;
        vals[totNx*(totNy-1) + totNx-1-SHIFT][2] = 0;
      }
    }
    // TODO fix last dof for periodic BC

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
    HYPRE_StructSMGSetTol(hypre_solver, 1e-3);
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
// let's relinquish STRIDE which was only added for clarity:
#undef STRIDE

HYPREdirichlet::~HYPREdirichlet()
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

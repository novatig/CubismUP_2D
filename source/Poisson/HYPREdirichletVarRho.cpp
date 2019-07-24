//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "HYPREdirichletVarRho.h"

using namespace cubism;

void HYPREdirichletVarRho::solve(const std::vector<BlockInfo>& BSRC,
                                 const std::vector<BlockInfo>& BDST)
{
  #ifdef HYPREFFT

  static constexpr double EPS = std::numeric_limits<double>::epsilon();
  const size_t nBlocks = BDST.size();
  HYPRE_Int ilower[2] = {0,0}, iupper[2] = {(int)totNx-1, (int)totNy-1};

  sim.startProfiler("HYPRE_cub2rhs");

  // pre-hypre solve plan:
  // 1) place initial guess of pressure into vector x
  // 2) in the same compute discrepancy from sum RHS = 0
  // 3) send initial guess for x to hypre
  // 4) correct RHS such that sum RHS = 0 due to boundary conditions
  // 5) give RHS to hypre
  // 6) if user modified matrix, make sure it respects neumann BC
  // 7) if user modified matrix, reassemble it so that hypre updates precond
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nBlocks; ++i)
    {
      const size_t blocki = VectorBlock::sizeX * BSRC[i].index[0];
      const size_t blockj = VectorBlock::sizeY * BSRC[i].index[1];
      const ScalarBlock& P = *(ScalarBlock*) BDST[i].ptrBlock;
      const size_t blockStart = blocki + stride * blockj;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        buffer[blockStart + ix + stride*iy] = P(ix,iy).s; // 1)
    }
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, buffer); // 3)

    Real sumRHS = 0, sumABS = 0;
    #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
    for(size_t i=0; i<nBlocks; ++i) {
      const size_t blocki = VectorBlock::sizeX * BSRC[i].index[0];
      const size_t blockj = VectorBlock::sizeY * BSRC[i].index[1];
      const ScalarBlock& RHS = *(ScalarBlock*) BSRC[i].ptrBlock;
      const size_t start = blocki + stride * blockj;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) { // 4)
        buffer[start +ix +stride*iy] = RHS(ix,iy).s;
        sumRHS +=           RHS(ix,iy).s;  // 2)
        sumABS += std::fabs(RHS(ix,iy).s); // 2)
      }
    }

    //printf("Relative RHS correction:%e\n", sumRHS/std::max(EPS,sumABS) );
    HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, buffer); // 5)
  }

  if(not bPeriodic && 1) // 6)
  {
    #if 0
     #pragma omp parallel for schedule(static)
     for (size_t i = 0; i < totNx; ++i) {
      // first south row
      matAry[stride*(0)       +i][3]  = 0; // south
      // first north row
      matAry[stride*(totNy-1) +i][4]  = 0; // north
     }
     #pragma omp parallel for schedule(static)
     for (size_t j = 0; j < totNy; ++j) {
      // first west col
      matAry[stride * j +      0][1]  = 0; // west
      // first east col
      matAry[stride * j +totNx-1][2]  = 0; // east
     }
    #else
     #pragma omp parallel for schedule(static)
     for (size_t i = 0; i < totNx; ++i) {
      // first south row
      matAry[stride*(0)       +i][0] += matAry[stride*(0)       +i][3]; // cc
      matAry[stride*(0)       +i][3]  = 0; // south
      // first north row
      matAry[stride*(totNy-1) +i][0] += matAry[stride*(totNy-1) +i][4]; // cc
      matAry[stride*(totNy-1) +i][4]  = 0; // north
     }
     #pragma omp parallel for schedule(static)
     for (size_t j = 0; j < totNy; ++j) {
      // first west col
      matAry[stride * j +      0][0] += matAry[stride*j +       0][1]; // center
      matAry[stride * j +      0][1]  = 0; // west
      // first east col
      matAry[stride * j +totNx-1][0] += matAry[stride*j + totNx-1][2]; // center
      matAry[stride * j +totNx-1][2]  = 0; // east
     }
    #endif
  }

  if(1) // 7)
  {
    Real * const linV = (Real*) matAry;
    // These indices must match to those in the offset array:
    HYPRE_Int inds[5] = {0, 1, 2, 3, 4};
    HYPRE_StructMatrixSetBoxValues(hypre_mat, ilower, iupper, 5, inds, linV);
    HYPRE_StructMatrixAssemble(hypre_mat);
  }

  sim.stopProfiler();

  if(0)
  {
    char fname[512]; sprintf(fname, "RHS_%06d", sim.step);
    sim.dumpTmp2( std::string(fname) );
  }

  sim.startProfiler("HYPRE_solve");
  if (solver == "gmres")
    HYPRE_StructLGMRESSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else if (solver == "bicgstab")
    HYPRE_StructBiCGSTABSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else if (solver == "pfmg")
    HYPRE_StructPFMGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else if (solver == "pcg")
    HYPRE_StructPCGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else HYPRE_StructHybridSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_getBoxV");
  HYPRE_StructVectorGetBoxValues(hypre_sol, ilower, iupper, buffer);
  sim.stopProfiler();

  sim.startProfiler("HYPRE_sol2cub");

  if(1) // remove mean pressure
  {
    Real avgP = 0;
    const Real fac = 1.0 / (totNx * totNy);
    #pragma omp parallel for schedule(static) reduction(+ : avgP)
    for (size_t i = 0; i < totNy*totNx; ++i) avgP += fac * buffer[i];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < totNy*totNx; ++i) buffer[i] -= avgP;
    printf("Average pressure:%e\n", avgP);
  }

  sol2cub(BDST);

  sim.stopProfiler();

  #endif
}

#define STRIDE s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX

HYPREdirichletVarRho::HYPREdirichletVarRho(SimulationData& s) :
  PoissonSolver(s, STRIDE), solver("gmres") //
{
  #ifdef HYPREFFT

  printf("Employing VarRho HYPRE-based Poisson solver with Dirichlet BCs.\n");
  buffer = new Real[totNy * totNx];
  HYPRE_Int ilower[2] = {0,0}, iupper[2] = {(int)totNx-1, (int)totNy-1};

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
    HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
    HYPRE_StructStencilCreate(2, 5, &hypre_stencil);
    for (int j = 0; j < 5; ++j)
      HYPRE_StructStencilSetElement(hypre_stencil, j, offsets[j]);
  }

  { // Matrix
    HYPRE_StructMatrixCreate(COMM, hypre_grid, hypre_stencil, &hypre_mat);
    HYPRE_StructMatrixInitialize(hypre_mat);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < totNy*totNx; ++i) {
      matAry[i][0] = -4;
      matAry[i][1] = 1; matAry[i][2] = 1; matAry[i][3] = 1; matAry[i][4] = 1;
    }
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
    HYPRE_StructLGMRESCreate(COMM, &hypre_solver);
    HYPRE_StructLGMRESSetTol(hypre_solver, 1e-3);
    HYPRE_StructLGMRESSetPrintLevel(hypre_solver, 1);
    HYPRE_StructLGMRESSetAbsoluteTol(hypre_solver, 1e-3);
    HYPRE_StructLGMRESSetMaxIter(hypre_solver, 50);
    HYPRE_StructLGMRESSetAugDim(hypre_solver, 2);
    HYPRE_StructLGMRESSetKDim(hypre_solver, 16);
    HYPRE_StructLGMRESSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else if (solver == "bicgstab") {
    printf("Using BiCGSTAB solver\n");
    HYPRE_StructBiCGSTABCreate(COMM, &hypre_solver);
    HYPRE_StructBiCGSTABSetMaxIter(hypre_solver, 50);
    HYPRE_StructBiCGSTABSetTol(hypre_solver, 1e-3);
    HYPRE_StructBiCGSTABSetAbsoluteTol(hypre_solver, 1e-3);
    HYPRE_StructBiCGSTABSetPrintLevel(hypre_solver, 0);
    if(0) { // Use SMG preconditioner: BAD
      HYPRE_StructPFMGCreate(COMM, &hypre_precond);
      HYPRE_StructPFMGSetMaxIter(hypre_precond, 50);
      HYPRE_StructPFMGSetTol(hypre_precond, 1e-3);
      HYPRE_StructPFMGSetRelChange(hypre_precond, 1e-3);
      HYPRE_StructPFMGSetPrintLevel(hypre_precond, 3);
      HYPRE_StructSMGSetNumPreRelax(hypre_precond, 1);
      HYPRE_StructSMGSetNumPostRelax(hypre_precond, 1);
      HYPRE_StructBiCGSTABSetPrecond(hypre_solver, HYPRE_StructPFMGSolve,
                                     HYPRE_StructPFMGSetup, hypre_precond);
    }
    HYPRE_StructBiCGSTABSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else if (solver == "pfmg") {
    printf("Using SMG solver\n");
    HYPRE_StructPFMGCreate(COMM, &hypre_solver);
    //HYPRE_StructSMGSetMemoryUse(hypre_solver, 0);
    HYPRE_StructPFMGSetMaxIter(hypre_solver, 100);
    HYPRE_StructPFMGSetTol(hypre_solver, 1e-3);
    HYPRE_StructPFMGSetRelChange(hypre_solver, 1e-4);
    HYPRE_StructPFMGSetPrintLevel(hypre_solver, 3);
    HYPRE_StructPFMGSetNumPreRelax(hypre_solver, 1);
    HYPRE_StructPFMGSetNumPostRelax(hypre_solver, 1);
    //HYPRE_StructSMGSetNonZeroGuess(hypre_solver);

    HYPRE_StructPFMGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else if (solver == "pcg") {
    printf("Using PCG solver\n");
    HYPRE_StructPCGCreate(COMM, &hypre_solver);
    HYPRE_StructPCGSetMaxIter(hypre_solver, 100);
    HYPRE_StructPCGSetTol(hypre_solver, 1e-4);
    HYPRE_StructPCGSetAbsoluteTol(hypre_solver, 1e-4);
    HYPRE_StructPCGSetPrintLevel(hypre_solver, 0);
    if(0) { // Use SMG preconditioner: BAD
      HYPRE_StructSMGCreate(COMM, &hypre_precond);
      HYPRE_StructSMGSetMaxIter(hypre_precond, 100);
      HYPRE_StructSMGSetTol(hypre_precond, 1e-3);
      HYPRE_StructSMGSetNumPreRelax(hypre_precond, 1);
      HYPRE_StructSMGSetNumPostRelax(hypre_precond, 1);
      HYPRE_StructPCGSetPrecond(hypre_solver, HYPRE_StructSMGSolve,
                                HYPRE_StructSMGSetup, hypre_precond);
    }
    HYPRE_StructPCGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else {
    printf("Using Hybrid solver\n");
    HYPRE_StructHybridCreate(COMM, &hypre_solver);
    HYPRE_StructHybridSetTol(hypre_solver, 1e-3);
    //HYPRE_StructHybridSetAbsoluteTol(hypre_solver, 1e-3);
    HYPRE_StructHybridSetConvergenceTol(hypre_solver, 1e-3);
    HYPRE_StructHybridSetDSCGMaxIter(hypre_solver, 50);
    HYPRE_StructHybridSetPCGMaxIter(hypre_solver, 50);
    HYPRE_StructHybridSetSolverType(hypre_solver, 2);
    //HYPRE_StructHybridSetKDim(hypre_solver, dim);
    HYPRE_StructHybridSetPrintLevel(hypre_solver, 0);
    HYPRE_StructHybridSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }

  #endif
}
// let's relinquish STRIDE which was only added for clarity:
#undef STRIDE

HYPREdirichletVarRho::~HYPREdirichletVarRho()
{
  #ifdef HYPREFFT

  if (solver == "gmres")
    HYPRE_StructLGMRESDestroy(hypre_solver);
  else if (solver == "bicgstab")
    HYPRE_StructBiCGSTABDestroy(hypre_solver);
  else if (solver == "pfmg")
    HYPRE_StructPFMGDestroy(hypre_solver);
  else if (solver == "pcg")
    HYPRE_StructPCGDestroy(hypre_solver);
  else HYPRE_StructHybridDestroy(hypre_solver);

  HYPRE_StructGridDestroy(hypre_grid);
  HYPRE_StructStencilDestroy(hypre_stencil);
  HYPRE_StructMatrixDestroy(hypre_mat);
  HYPRE_StructVectorDestroy(hypre_rhs);
  HYPRE_StructVectorDestroy(hypre_sol);
  delete [] buffer;
  delete [] matAry;

  #endif
}

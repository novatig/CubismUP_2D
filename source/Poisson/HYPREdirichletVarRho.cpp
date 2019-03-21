//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "HYPREdirichletVarRho.h"

#ifdef HYPREFFT

template<typename T>
static inline T mean(const T A, const T B)
{
  return 0.5*(A+B);
}

void HYPREdirichletVarRho::fadeoutBorder(const double dt) const
{
  static constexpr int Z = 8, buffer = 8;
  const Real h = sim.getH(), iWidth = 1/(buffer*h);
  const std::vector<BlockInfo>& velInfo  = sim.vel->getBlocksInfo();
  const Real extent[2] = {sim.bpdx/ (Real) std::max(sim.bpdx, sim.bpdy),
                          sim.bpdy/ (Real) std::max(sim.bpdx, sim.bpdy)};
  const auto _is_touching = [&] (const BlockInfo& i) {
    Real min_pos[2], max_pos[2];
    i.pos(max_pos, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    i.pos(min_pos, 0, 0);
    const bool touchN = (Z+buffer)*h >= extent[1] - max_pos[1];
    const bool touchE = (Z+buffer)*h >= extent[0] - max_pos[0];
    const bool touchS = (Z+buffer)*h >= min_pos[1];
    const bool touchW = (Z+buffer)*h >= min_pos[0];
    return touchN || touchE || touchS || touchW;
  };

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    if( not _is_touching(velInfo[i]) ) continue;
    const size_t blocki = VectorBlock::sizeX * velInfo.index[0];
    const size_t blockj = VectorBlock::sizeY * velInfo.index[1];
    const size_t blockStart = blocki + stride * blockj;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Real p[2];
      velInfo[i].pos(p, ix, iy);
      const Real arg1= std::max((Real)0, (Z+buffer)*h -(extent[0]-p[0]) );
      const Real arg2= std::max((Real)0, (Z+buffer)*h -(extent[1]-p[1]) );
      const Real arg3= std::max((Real)0, (Z+buffer)*h -p[0] );
      const Real arg4= std::max((Real)0, (Z+buffer)*h -p[1] );
      const Real dist= std::min(std::max({arg1, arg2, arg3, arg4}), buffer*h);
      const size_t idx = blockStart + ix + stride*iy;
      //RHS(ix, iy).s = std::max(1-factor, 1 - factor*std::pow(dist*iWidth, 2));
      dest[idx] *= 1 - std::pow(dist*iWidth, 2);
    }
  }
};

void HYPREdirichletVarRho::updateRHSandMAT(const double dt) const
{
  const std::vector<BlockInfo>& chiInfo  = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& velInfo  = sim.vel->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  RowType * __restrict__ const mat = matAry;
  Real * __restrict__ const dest = buffer;

  #pragma omp parallel
  {
    const Real h = sim.getH(), facDiv = 0.5*h/dt;
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab velLab;  velLab.prepare( *(sim.vel),    stenBeg, stenEnd, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const size_t blocki = VectorBlock::sizeX * velInfo.index[0];
      const size_t blockj = VectorBlock::sizeY * velInfo.index[1];
      const size_t blockStart = blocki + stride * blockj;
      velLab. load( velInfo[i], 0);
      uDefLab.load(uDefInfo[i], 0);
      iRhoLab.load(iRhoInfo[i], 0);
      const VectorLab  & __restrict__ V   =  velLab;
      const VectorLab  & __restrict__ UDEF= uDefLab;
      const ScalarLab  & __restrict__ IRHO= iRhoLab;
      const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)chiInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const size_t idx = blockStart + ix + stride*iy;
        const Real divVx  = V(ix+1,iy).u[0]    - V(ix-1,iy).u[0];
        const Real divVy  = V(ix,iy+1).u[1]    - V(ix,iy-1).u[1];
        const Real divUSx = UDEF(ix+1,iy).u[0] - UDEF(ix-1,iy).u[0];
        const Real divUSy = UDEF(ix,iy+1).u[1] - UDEF(ix,iy-1).u[1];
        dest[idx] = facDiv*( divVx+divVy - CHI(ix,iy).s*(divUSx+divUSy) );

        const Real coefN = mean(IRHO(ix+1,iy).s, IRHO(ix,iy).s);
        const Real coefS = mean(IRHO(ix-1,iy).s, IRHO(ix,iy).s);
        const Real coefE = mean(IRHO(ix,iy+1).s, IRHO(ix,iy).s);
        const Real coefW = mean(IRHO(ix,iy-1).s, IRHO(ix,iy).s);
        mat[idx][0] = -coefN -coefS -coefE -coefW;
        mat[idx][1] = coefW;
        mat[idx][2] = coefE;
        mat[idx][3] = coefS;
        mat[idx][4] = coefN;
      }
    }
  }

  fadeoutBorder();

  {
    Real sumRHS = 0, sumABS = 0;
    #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
    for (size_t i = 0; i < totNy*totNx; i++) {
      sumABS += std::fabs(dest[i]); sumRHS += dest[i];
    }
    sumABS = std::max(std::numeric_limits<Real>::epsilon(), sumABS);
    const Real corr = sumRHS / sumABS;
    printf("Relative RHS correction:%e\n", corr);
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<totNy*totNx; i++) dest[i] -= std::fabs(dest[i])*corr;
  }

  if(not bPeriodic) // 0 Neumann BC
  {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < totNx; i++) {
      // first south row
      mat[totNx*(0)       +i][0] += mat[totNx*(0)       +i][3]; // center
      mat[totNx*(0)       +i][3]  = 0; // south
      // first north row
      mat[totNx*(totNy-1) +i][0] += mat[totNx*(totNy-1) +i][4]; // center
      mat[totNx*(totNy-1) +i][4]  = 0; // north
    }
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < totNy; j++) {
      // first west col
      mat[totNx*j +       0][0] += mat[totNx*j +       0][1]; // center
      mat[totNx*j +       0][1]  = 0; // west
      // first east col
      mat[totNx*j + totNx-1][0] += mat[totNx*j + totNx-1][2]; // center
      mat[totNx*j + totNx-1][2]  = 0; // east
    }
  }
  {
    // set last corner such that last point has pressure pLast
    mat[totNx*(totNy-1) +totNx-1][1] = 0; // west
    mat[totNx*(totNy-1) +totNx-1][2] = 0; // east
    mat[totNx*(totNy-1) +totNx-1][3] = 0; // south
    mat[totNx*(totNy-1) +totNx-1][4] = 0; // north
    // preserve conditioning? P * coef = pLast * coef -> P = pLast
    dest[totNx*(totNy-1) +totNx-1]  = pLast * mat[totNx*(totNy-1) +totNx-1][0];
    // dof to the west and to the south get affected:
    dest[totNx*(totNy-2) +totNx-1] -= pLast * mat[totNx*(totNy-2) +totNx-1][4];
    mat [totNx*(totNy-2) +totNx-1][4] = 0;
    dest[totNx*(totNy-1) +totNx-2] -= pLast * mat[totNx*(totNy-1) +totNx-2][2];
    mat [totNx*(totNy-1) +totNx-2][2] = 0;
  }
  // TODO fix last dof for periodic BC

  Real * const linV = (Real*) matAry;
  // These indices must match to those in the offset array:
  HYPRE_Int inds[5] = {0, 1, 2, 3, 4};
  HYPRE_StructMatrixSetBoxValues(hypre_mat, ilower, iupper, 5, inds, linV);
  HYPRE_StructMatrixAssemble(hypre_mat);
}

void HYPREdirichletVarRho::solve(const std::vector<BlockInfo>& BSRC,
                                 const std::vector<BlockInfo>& BDST)
{
  HYPRE_Int ilower[2] = {0,0};
  HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};

  sim.startProfiler("HYPRE_cub2rhs");
  updateRHSandMAT(sim.dt);
  sim.stopProfiler();

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
  sol2cub(BDST);

  sim.stopProfiler();
}

#define STRIDE s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX

HYPREdirichletVarRho::HYPREdirichletVarRho(SimulationData& s) :
  PoissonSolver(s, STRIDE), solver("smg")
{
  printf("Employing VarRho HYPRE-based Poisson solver with Dirichlet BCs.\n");
  buffer = new Real[totNy * totNx];

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
    updateRHSandMAT(1.0);
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

HYPREdirichletVarRho::~HYPREdirichlet()
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
  delete [] vals;
}

#endif
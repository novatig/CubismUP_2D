//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "PressureIterator.h"
#include "HYPRE_solver.h"

Real PressureIterator::updatePenalizationForce(const double dt) const
{
  const Real h = sim.getH(), ffac = dt, pFac = -0.5*dt/h, lambda = 1/dt;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

  Real max_RelDforce = 0;
  #pragma omp parallel reduction(max : max_RelDforce)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
      const VectorBlock&__restrict__ VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock&__restrict__ UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            VectorBlock&__restrict__ TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock;
            VectorBlock&__restrict__    F= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real X = CHI(ix,iy).s;
        const Real pGradx = pFac * (P(ix+1,iy).s - P(ix-1,iy).s);
        const Real pGrady = pFac * (P(ix,iy+1).s - P(ix,iy-1).s);
        const Real fPenlx = ffac * F(ix,iy).u[0] * X;
        const Real fPenly = ffac * F(ix,iy).u[1] * X;
        const Real Unxt = VEL(ix,iy).u[0] + pGradx + fPenlx;
        const Real Vnxt = VEL(ix,iy).u[1] + pGrady + fPenly;
        const Real dFx = lambda * ( Unxt - UDEF(ix,iy).u[0] );
        const Real dFy = lambda * ( Vnxt - UDEF(ix,iy).u[1] );
        TMPV(ix,iy).u[0] = X * dFx;
        TMPV(ix,iy).u[1] = X * dFy;
        F(ix,iy).u[0] += dFx;
        F(ix,iy).u[1] += dFy;
        const Real fNorm = std::pow(F(ix,iy).u[0],2) +std::pow(F(ix,iy).u[1],2);
        const Real relErrFac = 1 / std::sqrt( EPS + X * X * fNorm );
        const Real err_x = std::fabs(X*dFx) * relErrFac;
        const Real err_y = std::fabs(X*dFy) * relErrFac;
        max_RelDforce = std::max(max_RelDforce, std::max(err_x, err_y));
      }
    }
  }

  return max_RelDforce;
}

void PressureIterator::finalizeVelocity(const double dt) const
{
  const Real h = sim.getH(), ffac = dt, pFac = -0.5*dt/h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
            VectorBlock&__restrict__ VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
      const VectorBlock&__restrict__ FPNL= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real pGradx = pFac * (P(ix+1,iy).s - P(ix-1,iy).s);
        const Real pGrady = pFac * (P(ix,iy+1).s - P(ix,iy-1).s);
        const Real fPenlx = ffac * FPNL(ix,iy).u[0] * CHI(ix,iy).s;
        const Real fPenly = ffac * FPNL(ix,iy).u[1] * CHI(ix,iy).s;
        VEL(ix,iy).u[0] += pGradx + fPenlx;
        VEL(ix,iy).u[1] += pGrady + fPenly;
      }
    }
  }
}

void PressureIterator::updatePressureRHS() const
{
  const Real h = sim.getH(), fac = 0.5/h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab tmplab; tmplab.prepare(*(sim.tmpV), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      tmplab.load(tmpVInfo[i], 0); // loads dPenal Force field with ghosts
      const VectorLab & __restrict__ dF = tmplab;
      ScalarBlock & __restrict__ pRHS = *(ScalarBlock*)pRHSInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        const Real d_dFx = dF(ix+1,iy).u[0] - dF(ix-1,iy).u[0];
        const Real d_dFy = dF(ix,iy+1).u[1] - dF(ix,iy-1).u[1];
        pRHS(ix,iy).s += fac*(d_dFx+d_dFy);
      }
    }
  }
}

void PressureIterator::operator()(const double dt)
{
  for(int iter = 0; iter < 100; iter++)
  {
    sim.startProfiler("PressureIterator::updatePenalizationForce");
    const Real max_RelDforce = updatePenalizationForce(dt);
    sim.stopProfiler();
    printf("iter:%02d - max relative error: %f\n", iter, max_RelDforce);

    if(max_RelDforce > 0.01)
    {
      sim.startProfiler("PressureIterator::updatePressureRHS");
      updatePressureRHS();
      sim.stopProfiler();

      pressureSolver->solve();
    }
    else
    {
      sim.startProfiler("PressureIterator::finalizeVelocity");
      finalizeVelocity(dt);
      sim.stopProfiler();
      break;
    }
  }
}

PressureIterator::PressureIterator(SimulationData& s) : Operator(s),
pressureSolver(new HYPRE_solver(sim)) {}

PressureIterator::~PressureIterator() {
  delete pressureSolver;
}

//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "PressureIterator.h"
#include "HYPRE_solver.h"

void PressureIterator::initPenalizationForce(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, invDt = 1/dt;

  #pragma omp parallel
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
        const Real Unxt = VEL(ix,iy).u[0] + pFac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real Vnxt = VEL(ix,iy).u[1] + pFac*(P(ix,iy+1).s - P(ix,iy-1).s);
        F(ix,iy).u[0] = CHI(ix,iy).s * invDt * ( UDEF(ix,iy).u[0] - Unxt );
        F(ix,iy).u[1] = CHI(ix,iy).s * invDt * ( UDEF(ix,iy).u[1] - Vnxt );
        TMPV(ix,iy).u[0] = VEL(ix,iy).u[0]; // copy vel before P and F onto temp
        TMPV(ix,iy).u[1] = VEL(ix,iy).u[1]; // copy vel before P and F onto temp
      }
    }
  }
}

Real PressureIterator::updatePenalizationForce(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, invDt = 1/dt;//, hsq = h*h;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

  Real MX = 0, MY = 0, EX = 0, EY = 0;
  #pragma omp parallel reduction(max : MX,MY,EX,EY)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
            VectorBlock&__restrict__ VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock&__restrict__ TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const VectorBlock&__restrict__ UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            VectorBlock&__restrict__    F= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real X = CHI(ix,iy).s;
        const Real Unxt = TMPV(ix,iy).u[0] + pFac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real Vnxt = TMPV(ix,iy).u[1] + pFac*(P(ix,iy+1).s - P(ix,iy-1).s);
        VEL(ix,iy).u[0] = Unxt + dt * F(ix,iy).u[0];
        VEL(ix,iy).u[1] = Vnxt + dt * F(ix,iy).u[1];
        F(ix,iy).u[0] = X * invDt *( UDEF(ix,iy).u[0] - Unxt );
        F(ix,iy).u[1] = X * invDt *( UDEF(ix,iy).u[1] - Vnxt );
        MX = std::max( MX, X * std::fabs(UDEF(ix,iy).u[0]                  ) );
        EX = std::max( EX, X * std::fabs(UDEF(ix,iy).u[0] - VEL(ix,iy).u[0]) );
        MY = std::max( MY, X * std::fabs(UDEF(ix,iy).u[1]                  ) );
        EY = std::max( EY, X * std::fabs(UDEF(ix,iy).u[1] - VEL(ix,iy).u[1]) );
      }
    }
  }
  cout <<EX<<" "<<EY<<" "<<MX<<" "<<MY<<endl;
  return std::max(EX,EY) / (EPS + std::max(MX,MY));
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
      const VectorBlock&__restrict__ FPNL= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real pGradx = pFac * (P(ix+1,iy).s - P(ix-1,iy).s);
        const Real pGrady = pFac * (P(ix,iy+1).s - P(ix,iy-1).s);
        const Real fPenlx = ffac * FPNL(ix,iy).u[0];
        const Real fPenly = ffac * FPNL(ix,iy).u[1];
        VEL(ix,iy).u[0] += pGradx + fPenlx;
        VEL(ix,iy).u[1] += pGrady + fPenly;
      }
    }
  }
}

void PressureIterator::updatePressureRHS() const
{
  const Real h = sim.getH(), facDiv = 0.5*h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab forcelab; forcelab.prepare(*(sim.force), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      forcelab.load(forceInfo[i], 0); // loads dPenal Force field with ghosts
      const VectorLab & __restrict__ F = forcelab;
      const ScalarBlock& __restrict__ pRHS =*(ScalarBlock*)pRHSInfo[i].ptrBlock;
            ScalarBlock& __restrict__ TMP  =*(ScalarBlock*) tmpInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        const Real d_Fx = F(ix+1,iy).u[0] - F(ix-1,iy).u[0];
        const Real d_Fy = F(ix,iy+1).u[1] - F(ix,iy-1).u[1];
        TMP(ix, iy).s = pRHS(ix,iy).s + facDiv*(d_Fx+d_Fy);
      }
    }
  }
}

void PressureIterator::operator()(const double dt)
{
  sim.startProfiler("PressureIterator_initPenalizationForce");
  initPenalizationForce(dt);
  sim.stopProfiler();

  for(int iter = 0; iter < 100; iter++)
  {
    sim.startProfiler("PressureIterator_updatePressureRHS");
    updatePressureRHS();
    sim.stopProfiler();

    pressureSolver->solve();
    sim.dumpPres("iter_"+std::to_string(iter)+"_");
    sim.startProfiler("PressureIterator_updatePenalizationForce");
    const Real max_RelDforce = updatePenalizationForce(dt);
    sim.stopProfiler();
    printf("iter:%02d - max relative error: %f\n", iter, max_RelDforce);
    if(max_RelDforce < 0.02) break;
  }
}

PressureIterator::PressureIterator(SimulationData& s) : Operator(s),
pressureSolver(new HYPRE_solver(sim)) {}

PressureIterator::~PressureIterator() {
  delete pressureSolver;
}

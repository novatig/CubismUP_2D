//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "PressureIterator.h"
#include "../Poisson/FFTW_freespace.h"
#include "../Poisson/HYPREdirichlet.h"

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
            VectorBlock&__restrict__ VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock&__restrict__ UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            VectorBlock&__restrict__ TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock;
            VectorBlock&__restrict__    F= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        TMPV(ix,iy).u[0] = VEL(ix,iy).u[0]; // copy vel before P and F onto temp
        TMPV(ix,iy).u[1] = VEL(ix,iy).u[1]; // copy vel before P and F onto temp
        const Real X= CHI(ix,iy).s, US= UDEF(ix,iy).u[0], VS= UDEF(ix,iy).u[1];
        const Real Unxt = VEL(ix,iy).u[0] + pFac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real Vnxt = VEL(ix,iy).u[1] + pFac*(P(ix,iy+1).s - P(ix,iy-1).s);
        #ifdef SOFT_PENL
          const Real uTgt = (Unxt + X*US)/(1+X), vTgt = (Vnxt + X*VS)/(1+X);
        #else
          const Real uTgt = US, vTgt = VS;
        #endif
        F(ix,iy).u[0] = X * invDt * ( uTgt - Unxt );
        F(ix,iy).u[1] = X * invDt * ( vTgt - Vnxt );
        VEL(ix,iy).u[0] = Unxt + dt * F(ix,iy).u[0];
        VEL(ix,iy).u[1] = Vnxt + dt * F(ix,iy).u[1];
      }
    }
  }
}

Real PressureIterator::updatePenalizationForce(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, invDt = 1/dt;//, hsq = h*h;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

  // If in debug we also make sure that flow velocity inside the obstacle
  // Is exactly the same as obstacle's velocity. It should always be up to EPS.
  #ifndef NDEBUG
    Real Vx = 0, Vy = 0, dVx = 0, dVy = 0;
  #endif
  // Measure how much F has changed between iterations to check convergence:
  Real Fx = 0, Fy = 0, dFx = 0, dFy = 0;

  #ifndef NDEBUG
  #pragma omp parallel reduction(max : Fx, Fy, dFx, dFy, Vx, Vy, dVx, dVy)
  #else
  #pragma omp parallel reduction(max : Fx, Fy, dFx, dFy)
  #endif
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
      //    ScalarBlock&__restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            VectorBlock&__restrict__    F= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real US= UDEF(ix,iy).u[0], VS= UDEF(ix,iy).u[1], X= CHI(ix,iy).s;
        const Real Unxt = TMPV(ix,iy).u[0] + pFac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real Vnxt = TMPV(ix,iy).u[1] + pFac*(P(ix,iy+1).s - P(ix,iy-1).s);
        #ifdef SOFT_PENL
          const Real uTgt = (Unxt + X*US)/(1+X), vTgt = (Vnxt + X*VS)/(1+X);
        #else
          const Real uTgt = US, vTgt = VS;
        #endif
        const Real FXnxt = X*invDt*(uTgt-Unxt), FYnxt = X*invDt*(vTgt-Vnxt);
        Fx  = std::max(  Fx, std::fabs(FXnxt                ) );
        Fy  = std::max(  Fy, std::fabs(FYnxt                ) );
        dFx = std::max( dFx, std::fabs(FXnxt - F(ix,iy).u[0]) );
        dFy = std::max( dFy, std::fabs(FYnxt - F(ix,iy).u[1]) );

        F(ix,iy).u[0] = FXnxt; VEL(ix,iy).u[0] = Unxt + dt * FXnxt;
        F(ix,iy).u[1] = FYnxt; VEL(ix,iy).u[1] = Vnxt + dt * FYnxt;

        #ifndef NDEBUG
          Vx  = std::max( Vx, ((1-X)<EPS) * std::fabs(uTgt                ) );
          Vy  = std::max( Vy, ((1-X)<EPS) * std::fabs(vTgt                ) );
          dVx = std::max(dVx, ((1-X)<EPS) * std::fabs(uTgt-VEL(ix,iy).u[0]) );
          dVy = std::max(dVy, ((1-X)<EPS) * std::fabs(vTgt-VEL(ix,iy).u[1]) );
        #endif
      }
    }
  }
  // By definition of the eqns, velocity should be up to eps equal to imposed:
  assert(std::max(dVx,dVy) / (EPS + std::max(Vx,Vy)) < std::sqrt(EPS));
  return std::max(dFx,dFy) / (EPS + std::max(Fx,Fy));
}

void PressureIterator::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*h, invDt = 1/dt;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab forcelab; forcelab.prepare(*(sim.force), stenBeg, stenEnd, 0);
    ScalarLab chilab;     chilab.prepare(*(sim.chi), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      forcelab.load(forceInfo[i], 0); // loads dPenal Force field with ghosts
      chilab.load(chiInfo[i], 0); // loads chi field with ghosts
      const VectorLab  & __restrict__ F = forcelab;
      const ScalarLab  & __restrict__ CHI = chilab; // only this needs ghosts
      const ScalarBlock& __restrict__ pRHS =*(ScalarBlock*)pRHSInfo[i].ptrBlock;
            ScalarBlock& __restrict__ TMP  =*(ScalarBlock*) tmpInfo[i].ptrBlock;
      const VectorBlock& __restrict__ VEL  =*(VectorBlock*) velInfo[i].ptrBlock;
      const VectorBlock& __restrict__ UDEF =*(VectorBlock*)uDefInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        const Real dChix = facDiv*(CHI(ix+1,iy).s - CHI(ix-1,iy).s);
        const Real dChiy = facDiv*(CHI(ix,iy+1).s - CHI(ix,iy-1).s);
        const Real dUbndX = invDt *(UDEF(ix,iy).u[0] - VEL(ix,iy).u[0]);
        const Real dUbndY = invDt *(UDEF(ix,iy).u[1] - VEL(ix,iy).u[1]);
        const Real d_Fx = facDiv *(F(ix+1,iy).u[0] - F(ix-1,iy).u[0]);
        const Real d_Fy = facDiv *(F(ix,iy+1).u[1] - F(ix,iy-1).u[1]);
        TMP(ix, iy).s = pRHS(ix,iy).s + (d_Fx+d_Fy) -dChix*dUbndX -dChiy*dUbndY;
      }
    }
  }
}

void PressureIterator::operator()(const double dt)
{
  sim.startProfiler("PIter_init");
  initPenalizationForce(dt);
  sim.stopProfiler();

  for(int iter = 0; iter < 100; iter++)
  {
    sim.startProfiler("PIter_RHS");
    updatePressureRHS(dt);
    sim.stopProfiler();

    pressureSolver->solve();
    //sim.dumpPres("iter_"+std::to_string(iter)+"_");
    sim.startProfiler("PIter_update");
    const Real max_RelDforce = updatePenalizationForce(dt);
    //sim.dumpTmp("iter_"+std::to_string(iter)+"_");
    sim.stopProfiler();
    printf("iter:%02d - max relative error: %f\n", iter, max_RelDforce);
    if(max_RelDforce < 0.01) break;
  }
}

PressureIterator::PressureIterator(SimulationData& s) : Operator(s),
pressureSolver(
  s.poissonType=="hypre"? static_cast<PoissonSolver*>(new HYPREdirichlet(sim))
                        : static_cast<PoissonSolver*>(new FFTW_freespace(sim)) )
{
}

PressureIterator::~PressureIterator() {
  delete pressureSolver;
}

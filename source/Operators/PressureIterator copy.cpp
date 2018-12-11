//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "PressureIterator.h"
#include "../Poisson/HYPRE_solver.h"
#include "../Poisson/PoissonSolverScalarFFTW_freespace.h"

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
        const Real Unxt = VEL(ix,iy).u[0] + pFac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real Vnxt = VEL(ix,iy).u[1] + pFac*(P(ix,iy+1).s - P(ix,iy-1).s);
        const Real uTgt = (Unxt+CHI(ix,iy).s*UDEF(ix,iy).u[0])/(1+CHI(ix,iy).s);
        const Real vTgt = (Vnxt+CHI(ix,iy).s*UDEF(ix,iy).u[1])/(1+CHI(ix,iy).s);

        F(ix,iy).u[0] = CHI(ix,iy).s * invDt * ( UDEF(ix,iy).u[0] - Unxt );
        F(ix,iy).u[1] = CHI(ix,iy).s * invDt * ( UDEF(ix,iy).u[1] - Vnxt );
        TMPV(ix,iy).u[0] = VEL(ix,iy).u[0]; // copy vel before P and F onto temp
        TMPV(ix,iy).u[1] = VEL(ix,iy).u[1]; // copy vel before P and F onto temp
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
      //    ScalarBlock&__restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            VectorBlock&__restrict__    F= *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real X = CHI(ix,iy).s, IN = (1-X) < EPS;
        const Real Unxt = TMPV(ix,iy).u[0] + pFac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real Vnxt = TMPV(ix,iy).u[1] + pFac*(P(ix,iy+1).s - P(ix,iy-1).s);
        VEL(ix,iy).u[0] = Unxt + dt * F(ix,iy).u[0];
        VEL(ix,iy).u[1] = Vnxt + dt * F(ix,iy).u[1];
        F(ix,iy).u[0] = X * invDt *( UDEF(ix,iy).u[0] - Unxt );
        F(ix,iy).u[1] = X * invDt *( UDEF(ix,iy).u[1] - Vnxt );
        //TMP(ix,iy).s = X * (UDEF(ix,iy).u[0] - VEL(ix,iy).u[0]);
        MX = std::max( MX, IN * std::fabs(UDEF(ix,iy).u[0]                  ) );
        EX = std::max( EX, IN * std::fabs(UDEF(ix,iy).u[0] - VEL(ix,iy).u[0]) );
        MY = std::max( MY, IN * std::fabs(UDEF(ix,iy).u[1]                  ) );
        EY = std::max( EY, IN * std::fabs(UDEF(ix,iy).u[1] - VEL(ix,iy).u[1]) );
      }
    }
  }
  cout <<EX<<" "<<EY<<" "<<MX<<" "<<MY<<endl;
  return std::max(EX,EY) / (EPS + std::max(MX,MY));
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
  sim.startProfiler("PressureIterator_initPenalizationForce");
  initPenalizationForce(dt);
  sim.stopProfiler();

  for(int iter = 0; iter < 100; iter++)
  {
    sim.startProfiler("PressureIterator_updatePressureRHS");
    updatePressureRHS(dt);
    sim.stopProfiler();

    fftwSolver->solve();
    //pressureSolver->solve();
    //sim.dumpPres("iter_"+std::to_string(iter)+"_");
    sim.startProfiler("PressureIterator_updatePenalizationForce");
    const Real max_RelDforce = updatePenalizationForce(dt);
    //sim.dumpTmp("iter_"+std::to_string(iter)+"_");
    sim.stopProfiler();
    printf("iter:%02d - max relative error: %f\n", iter, max_RelDforce);
    if(max_RelDforce < 0.01) break;
  }
}

PressureIterator::PressureIterator(SimulationData& s) : Operator(s),
pressureSolver(new HYPRE_solver(sim)),
fftwSolver(new PoissonSolverFreespace(sim)) {}

PressureIterator::~PressureIterator() {
  delete pressureSolver;
}

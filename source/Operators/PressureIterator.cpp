//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureIterator.h"
#include "../Poisson/FFTW_freespace.h"
#include "../Poisson/HYPREdirichlet.h"
#include "../Poisson/FFTW_dirichlet.h"
#include "../Poisson/FFTW_periodic.h"
#ifdef CUDAFFT
#include "../Poisson/CUDA_all.h"
#endif
#include "Utils/BufferedLogger.h"

static inline PoissonSolver * makeSolver(SimulationData& sim)
{
  if (sim.poissonType == "hypre")
    return static_cast<PoissonSolver*>(new HYPREdirichlet(sim));
  else
  if (sim.poissonType == "periodic")
    return static_cast<PoissonSolver*>(new FFTW_periodic(sim));
  else
  if (sim.poissonType == "cosine")
    return static_cast<PoissonSolver*>(new FFTW_dirichlet(sim));
  #ifdef CUDAFFT
  else
  if (sim.poissonType == "cuda-periodic")
    return static_cast<PoissonSolver*>(new CUDA_periodic(sim));
  else
  if (sim.poissonType == "cuda-freespace")
    return static_cast<PoissonSolver*>(new CUDA_freespace(sim));
  #endif
  else
    return static_cast<PoissonSolver*>(new FFTW_freespace(sim));
}

//#define SOFT_PENL

void PressureIterator::initPenalizationForce(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, invDt = 1/dt;//sim.lambda;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock&__restrict__  US = *(VectorBlock*) uDefInfo[i].ptrBlock;
      const ScalarBlock&__restrict__   X = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            VectorBlock&__restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
            VectorBlock&__restrict__   F = *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        TMP(ix,iy).u[0] = V(ix,iy).u[0];
        TMP(ix,iy).u[1] = V(ix,iy).u[1];
        const Real Ufluid = V(ix,iy).u[0] + pFac * (P(ix+1,iy).s-P(ix-1,iy).s);
        const Real Vfluid = V(ix,iy).u[1] + pFac * (P(ix,iy+1).s-P(ix,iy-1).s);
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] = Ufluid + dt * F(ix,iy).u[0];
        V(ix,iy).u[1] = Vfluid + dt * F(ix,iy).u[1];
        const Real CHI = X(ix,iy).s, USX = US(ix,iy).u[0], USY = US(ix,iy).u[1];
        #if 0
          #ifdef SOFT_PENL
            const Real uTgt = (Ufluid + CHI*USX)/(1+CHI);
            const Real vTgt = (Vfluid + CHI*USY)/(1+CHI);
          #else
            const Real uTgt = CHI*USX + (1-CHI)*Ufluid;
            const Real vTgt = CHI*USY + (1-CHI)*Vfluid;
          #endif
          const Real dFx = CHI * invDt * (uTgt - V(ix,iy).u[0]);
          const Real dFy = CHI * invDt * (vTgt - V(ix,iy).u[1]);
          F(ix,iy).u[0] += dFx; F(ix,iy).u[1] += dFy;
        #else
          F(ix,iy).u[0] = CHI * invDt * (USX - V(ix,iy).u[0]);
          F(ix,iy).u[1] = CHI * invDt * (USY - V(ix,iy).u[1]);
        #endif
      }
    }
  }
}

Real PressureIterator::updatePenalizationForce(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, invDt = 1/dt;//sim.lambda;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

  // Measure how much F has changed between iterations to check convergence:
  Real sumFx = 0, sumFy = 0, sumdFx = 0, sumdFy = 0;
  #pragma omp parallel reduction( + : sumFx, sumFy, sumdFx, sumdFy )
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
            VectorBlock&__restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const VectorBlock&__restrict__  US = *(VectorBlock*) uDefInfo[i].ptrBlock;
      const ScalarBlock&__restrict__   X = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            VectorBlock&__restrict__   F = *(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real Ufluid = TMP(ix,iy).u[0] + pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
        const Real Vfluid = TMP(ix,iy).u[1] + pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] = Ufluid + dt * F(ix,iy).u[0];
        V(ix,iy).u[1] = Vfluid + dt * F(ix,iy).u[1];
        #ifdef SOFT_PENL
          const Real uTgt = (Ufluid + X(ix,iy).s*US(ix,iy).u[0])/(1+X(ix,iy).s);
          const Real vTgt = (Vfluid + X(ix,iy).s*US(ix,iy).u[1])/(1+X(ix,iy).s);
        #else
          const Real uTgt = X(ix,iy).s*US(ix,iy).u[0] + (1-X(ix,iy).s)*Ufluid;
          const Real vTgt = X(ix,iy).s*US(ix,iy).u[1] + (1-X(ix,iy).s)*Vfluid;
        #endif
        const Real dFx = invDt * (uTgt - V(ix,iy).u[0]); //X(ix,iy).s *
        const Real dFy = invDt * (vTgt - V(ix,iy).u[1]); //X(ix,iy).s *
        F(ix,iy).u[0] += dFx; F(ix,iy).u[1] += dFy;
        sumFx += std::pow(F(ix,iy).u[0], 2); sumdFx += std::pow(dFx,2);
        sumFy += std::pow(F(ix,iy).u[1], 2); sumdFy += std::pow(dFy,2);
      }
    }
  }

  //assert(std::max(dVx,dVy) / (EPS + std::max(Vx,Vy)) < std::sqrt(EPS));
  //return std::max(dFx,dFy) / (EPS + std::max(Fx,Fy));
  return std::sqrt((sumdFx + sumdFy) / (EPS + sumFx + sumFy));
}

void PressureIterator::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*h;//, invDt = 1/dt;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab forcelab; forcelab.prepare(*(sim.force), stenBeg, stenEnd, 0);
    //ScalarLab chilab;     chilab.prepare(*(sim.chi), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      forcelab.load(forceInfo[i], 0); // loads dPenal Force field with ghosts
      //chilab.load(chiInfo[i], 0); // loads chi field with ghosts
      const VectorLab  & __restrict__ F = forcelab;
      //const ScalarLab  & __restrict__ CHI = chilab; // only this needs ghosts
      const ScalarBlock& __restrict__ pRHS =*(ScalarBlock*)pRHSInfo[i].ptrBlock;
            ScalarBlock& __restrict__ TMP  =*(ScalarBlock*) tmpInfo[i].ptrBlock;
      //const VectorBlock& __restrict__ VEL  =*(VectorBlock*) velInfo[i].ptrBlock;
      //const VectorBlock& __restrict__ UDEF =*(VectorBlock*)uDefInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        //const Real dChix = facDiv*(CHI(ix+1,iy).s - CHI(ix-1,iy).s);
        //const Real dChiy = facDiv*(CHI(ix,iy+1).s - CHI(ix,iy-1).s);
        const Real divFx = facDiv*(F(ix+1,iy).u[0] - F(ix-1,iy).u[0]);
        const Real divFy = facDiv*(F(ix,iy+1).u[1] - F(ix,iy-1).u[1]);
        //const Real dUbndX = invDt*(UDEF(ix,iy).u[0] - VEL(ix,iy).u[0]);
        //const Real dUbndY = invDt*(UDEF(ix,iy).u[1] - VEL(ix,iy).u[1]);
        TMP(ix,iy).s = pRHS(ix,iy).s +divFx +divFy;//-dChix*dUbndX-dChiy*dUbndY;
      }
    }
  }
}

void PressureIterator::operator()(const double dt)
{
  sim.startProfiler("PIter_init");
  initPenalizationForce(dt);
  sim.stopProfiler();

  int iter=0; Real relDF = 1e3;
  for(iter = 0; iter < 100; iter++)
  {
    sim.startProfiler("PIter_RHS");
    updatePressureRHS(dt);
    fadeoutBorder(dt);
    sim.stopProfiler();

    pressureSolver->iter = iter;
    pressureSolver->solve();

    //sim.dumpPres("iter_"+std::to_string(iter)+"_");
    sim.startProfiler("PIter_update");
    relDF = updatePenalizationForce(dt);
    //sim.dumpTmp("iter_"+std::to_string(iter)+"_");
    sim.stopProfiler();
    printf("iter:%02d - max relative error: %f\n", iter, relDF);
    if(relDF < 0.002) break;
  }

  if(not sim.muteAll)
  {
  std::stringstream ssF; ssF<<sim.path2file<<"/pressureIterStats.dat";
  std::ofstream pfile(ssF.str().c_str(), std::ofstream::app);
  if(sim.step==0) pfile<<"step time dt iter relDF"<<std::endl;
  pfile<<sim.step<<" "<<sim.time<<" "<<sim.dt<<" "<<iter<<" "<<relDF<<std::endl;
  }
}

PressureIterator::PressureIterator(SimulationData& s) : Operator(s),
pressureSolver( makeSolver(s) ) { }

PressureIterator::~PressureIterator() {
  delete pressureSolver;
}

void PressureIterator::fadeoutBorder(const double dt) const
{
  static constexpr int Z = 8, buffer = 8;
  const Real h = sim.getH(), iWidth = 1/(buffer*h);
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
    if( not _is_touching(tmpInfo[i]) ) continue;
    ScalarBlock& __restrict__ RHS = *(ScalarBlock*)tmpInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Real p[2];
      tmpInfo[i].pos(p, ix, iy);
      const Real arg1= std::max((Real)0, (Z+buffer)*h -(extent[0]-p[0]) );
      const Real arg2= std::max((Real)0, (Z+buffer)*h -(extent[1]-p[1]) );
      const Real arg3= std::max((Real)0, (Z+buffer)*h -p[0] );
      const Real arg4= std::max((Real)0, (Z+buffer)*h -p[1] );
      const Real dist= std::min(std::max({arg1, arg2, arg3, arg4}), buffer*h);
      //RHS(ix, iy).s = std::max(1-factor, 1 - factor*std::pow(dist*iWidth, 2));
      RHS(ix, iy).s *= 1 - std::pow(dist*iWidth, 2);
    }
  }
};

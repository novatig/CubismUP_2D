//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureSingle.h"
#include "../Poisson/FFTW_freespace.h"
#include "../Poisson/HYPREdirichlet.h"
#include "../Poisson/FFTW_dirichlet.h"
#include "../Poisson/FFTW_periodic.h"
#ifdef CUDAFFT
#include "../Poisson/CUDA_all.h"
#endif

#define SOFT_PENL

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

void PressureSingle::fadeoutBorder(const double dt) const
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

void PressureSingle::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*h/dt;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      vellab.load(velInfo[i], 0);
      const VectorLab  & __restrict__ V = vellab;
            ScalarBlock& __restrict__ TMP = *(ScalarBlock*)tmpInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real divVx = facDiv*(V(ix+1,iy).u[0] - V(ix-1,iy).u[0]);
        const Real divVy = facDiv*(V(ix,iy+1).u[1] - V(ix,iy-1).u[1]);
        TMP(ix, iy).s = divVx + divVy;
      }
    }
  }
}

void PressureSingle::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h;//, invDt = 1/dt;//sim.lambda;

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

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real Ufluid = V(ix,iy).u[0] + pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
        const Real Vfluid = V(ix,iy).u[1] + pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] = ( Ufluid + X(ix,iy).s*US(ix,iy).u[0] )/(1+X(ix,iy).s);
        V(ix,iy).u[1] = ( Vfluid + X(ix,iy).s*US(ix,iy).u[1] )/(1+X(ix,iy).s);
        //V(ix,iy).u[0] =  Ufluid;
        //V(ix,iy).u[1] =  Vfluid;
      }
    }
  }
}

void PressureSingle::operator()(const double dt)
{
  sim.startProfiler("PCorrect");
  updatePressureRHS(dt);
  fadeoutBorder(dt);
  sim.stopProfiler();

  pressureSolver->solve();

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureSingle::PressureSingle(SimulationData& s) : Operator(s),
pressureSolver( makeSolver(s) ) { }

PressureSingle::~PressureSingle() {
  delete pressureSolver;
}

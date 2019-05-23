//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureSingleStaggered.h"
#include "../Poisson/PoissonSolver.h"

using namespace cubism;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PressureSingleStaggered::fadeoutBorder(const double dt) const
{
  const Real fadeLenX = sim.fadeLenX, fadeLenY = sim.fadeLenY;
  const Real invFadeX = 1/(fadeLenX+EPS), invFadeY = 1/(fadeLenY+EPS);
  const auto& extent = sim.extents;
  const auto _is_touching = [&] (const BlockInfo& i) {
    Real min_pos[2], max_pos[2];
    i.pos(max_pos, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    i.pos(min_pos, 0, 0);
    const bool touchW = fadeLenX >= min_pos[0];
    const bool touchE = fadeLenX >= extent[0] - max_pos[0];
    const bool touchS = fadeLenY >= min_pos[1];
    const bool touchN = fadeLenY >= extent[1] - max_pos[1];
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
      Real p[2]; tmpInfo[i].pos(p, ix, iy);
      const Real yt = invFadeY*std::max(Real(0), fadeLenY - extent[1] + p[1] );
      const Real yb = invFadeY*std::max(Real(0), fadeLenY - p[1] );
      const Real xt = invFadeX*std::max(Real(0), fadeLenX - extent[0] + p[0] );
      const Real xb = invFadeX*std::max(Real(0), fadeLenX - p[0] );
      RHS(ix,iy).s *= 1-std::pow(std::min(std::max({yt,yb,xt,xb}), (Real)1), 2);
    }
  }
};

void PressureSingleStaggered::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = h/dt;

  #pragma omp parallel
  {
    static constexpr int stenBegV[3] = { 0, 0, 0}, stenEndV[3] = {2, 2, 1};
    static constexpr int stenBeg [3] = {-1,-1, 0}, stenEnd [3] = {2, 2, 1};
    VectorLab velLab;  velLab.prepare( *(sim.vel),  stenBegV, stenEndV, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      velLab.load( velInfo[i], 0); const auto & __restrict__ V   = velLab;
     uDefLab.load(uDefInfo[i], 0); const auto & __restrict__ UDEF= uDefLab;
      const auto& __restrict__ CHI = *(ScalarBlock*)chiInfo[i].ptrBlock;
            auto& __restrict__ TMP = *(ScalarBlock*)tmpInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real divVx  =    V(ix+1,iy).u[0] -    V(ix,iy).u[0];
        const Real divVy  =    V(ix,iy+1).u[1] -    V(ix,iy).u[1];
        const Real UDEFW = (UDEF(ix  ,iy).u[0] + UDEF(ix-1,iy).u[0]) / 2;
        const Real UDEFE = (UDEF(ix+1,iy).u[0] + UDEF(ix  ,iy).u[0]) / 2;
        const Real VDEFS = (UDEF(ix,iy  ).u[1] + UDEF(ix,iy-1).u[1]) / 2;
        const Real VDEFN = (UDEF(ix,iy+1).u[1] + UDEF(ix,iy  ).u[1]) / 2;
        const Real divUSx = UDEFE - UDEFW,  divUSy = VDEFN - VDEFS;
        TMP(ix, iy).s = facDiv*(divVx+divVy - CHI(ix,iy).s*(divUSx+divUSy));
      }
    }
  }
}

void PressureSingleStaggered::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h;
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 1, 1, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); const auto&__restrict__   P = plab;
      VectorBlock&__restrict__ V = *(VectorBlock*) velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] = V(ix,iy).u[0] + pFac *(P(ix,iy).s-P(ix-1,iy).s);
        V(ix,iy).u[1] = V(ix,iy).u[1] + pFac *(P(ix,iy).s-P(ix,iy-1).s);
      }
    }
  }
}

void PressureSingleStaggered::operator()(const double dt)
{
  sim.startProfiler("Prhs");
  updatePressureRHS(dt);
  fadeoutBorder(dt);
  sim.stopProfiler();

  pressureSolver->solve(tmpInfo, presInfo);

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureSingleStaggered::PressureSingleStaggered(SimulationData& s) : Operator(s),
pressureSolver( PoissonSolver::makeSolver(s) ) { }

PressureSingleStaggered::~PressureSingleStaggered() {
  delete pressureSolver;
}

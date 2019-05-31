//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureSingle.h"
#include "../Poisson/PoissonSolver.h"
#include "../Shape.h"

using namespace cubism;
using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PressureSingle::fadeoutBorder(const double dt) const
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

void PressureSingle::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*h/dt;
  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
  const size_t nShapes = sim.shapes.size();
  Real * sumRHS, * absRHS;
  sumRHS = (Real*) calloc(nShapes, sizeof(Real));
  absRHS = (Real*) calloc(nShapes, sizeof(Real));

  #pragma omp parallel reduction(+: sumRHS[:nShapes], absRHS[:nShapes])
  {
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      ( (ScalarBlock*)   tmpInfo[i].ptrBlock )->clear();
    for (size_t j=0; j < nShapes; j++)
    {
      const Shape * const shape = sim.shapes[j];
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      const ObstacleBlock*const o = OBLOCK[uDefInfo[i].blockID];
      if (o == nullptr) continue;

      uDefLab.load(uDefInfo[i], 0);
      const VectorLab  & __restrict__ UDEF= uDefLab;
      const CHI_MAT & __restrict__ chi = o->chi;
      ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if (chi[iy][ix] <= 0) continue;
        if(CHI(ix,iy).s > chi[iy][ix]) continue;
        const Real divUSx = UDEF(ix+1,iy).u[0] - UDEF(ix-1,iy).u[0];
        const Real divUSy = UDEF(ix,iy+1).u[1] - UDEF(ix,iy-1).u[1];
        const Real udefSrc = facDiv * chi[iy][ix] * (divUSx + divUSy);
        sumRHS[j] += udefSrc; absRHS[j] += std::fabs(udefSrc);
        TMP(ix, iy).s += facDiv * udefSrc;
      }
    }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  for (size_t j=0; j < nShapes; j++)
  {
    const Shape * const shape = sim.shapes[j];
    const Real corr = sumRHS[j] / std::max(absRHS[j], EPS);
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[uDefInfo[i].blockID];
    if (o == nullptr) continue;

    const CHI_MAT & __restrict__ chi = o->chi;
    ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      if(CHI(ix,iy).s > chi[iy][ix]) continue;
      TMP(ix, iy).s -= corr*std::fabs(TMP(ix, iy).s);
    }
  }

  free (sumRHS); free (absRHS);

  #pragma omp parallel
  {
    VectorLab velLab;  velLab.prepare( *(sim.vel),  stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      velLab.load(velInfo[i], 0); const VectorLab  & __restrict__ V   = velLab;
            ScalarBlock& __restrict__ TMP = *(ScalarBlock*)tmpInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real divVx  = V(ix+1,iy).u[0]    - V(ix-1,iy).u[0];
        const Real divVy  = V(ix,iy+1).u[1]    - V(ix,iy-1).u[1];
        TMP(ix, iy).s += facDiv*(divVx+divVy);
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

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] = V(ix,iy).u[0] + pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
        V(ix,iy).u[1] = V(ix,iy).u[1] + pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
      }
    }
  }
}

void PressureSingle::operator()(const double dt)
{
  sim.startProfiler("Prhs");
  updatePressureRHS(dt);
  //fadeoutBorder(dt);
  sim.stopProfiler();

  pressureSolver->solve(tmpInfo, presInfo);

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureSingle::PressureSingle(SimulationData& s) : Operator(s),
pressureSolver( PoissonSolver::makeSolver(s) ) { }

PressureSingle::~PressureSingle() {
  delete pressureSolver;
}

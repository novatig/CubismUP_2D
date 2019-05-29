//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureVarRho_proper.h"
#include "../Poisson/HYPREdirichletVarRho.h"

using namespace cubism;
static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PressureVarRho_proper::fadeoutBorder(const double dt) const
{
  const auto& extent = sim.extents;
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const Real fadeLenX = sim.fadeLenX, fadeLenY = sim.fadeLenY;
  const Real invFadeX = 1/(fadeLenX+EPS), invFadeY = 1/(fadeLenY+EPS);
  const auto _is_touching = [&] (const BlockInfo& i) {
    Real min_pos[2], max_pos[2]; i.pos(min_pos, 0, 0);
    i.pos(max_pos, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    const bool touchW = fadeLenX >= min_pos[0];
    const bool touchE = fadeLenX >= extent[0] - max_pos[0];
    const bool touchS = fadeLenY >= min_pos[1];
    const bool touchN = fadeLenY >= extent[1] - max_pos[1];
    return touchN || touchE || touchS || touchW;
  };

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++) {
    if( not _is_touching(tmpInfo[i]) ) continue;
    auto& __restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
    auto& __restrict__ RHS = *(ScalarBlock*) pRhsInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      Real p[2]; tmpInfo[i].pos(p, ix, iy);
      const Real yt = invFadeY*std::max(Real(0), fadeLenY - extent[1] + p[1] );
      const Real yb = invFadeY*std::max(Real(0), fadeLenY - p[1] );
      const Real xt = invFadeX*std::max(Real(0), fadeLenX - extent[0] + p[0] );
      const Real xb = invFadeX*std::max(Real(0), fadeLenX - p[0] );
      const Real fadeArg = std::min( std::max({yt, yb, xt, xb}), (Real)1 );
      RHS(ix,iy).s *= 1 - std::pow(fadeArg, 2);
      TMP(ix,iy).s *= 1 - std::pow(fadeArg, 2);
    }
  }
};

void PressureVarRho_proper::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = h/dt;
  const size_t stride = varRhoSolver->stride;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  HYPREdirichletVarRho::RowType* const mat = varRhoSolver->matAry;
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };

  #pragma omp parallel
  {
    static constexpr int stenBegV[3] = { 0, 0, 0}, stenEndV[3] = { 2, 2, 1};
    static constexpr int stenBeg [3] = {-1,-1, 0}, stenEnd [3] = { 2, 2, 1};
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);
    VectorLab velLab;   velLab.prepare(*(sim.vel), stenBegV, stenEndV, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const size_t blocki = VectorBlock::sizeX * velInfo[i].index[0];
      const size_t blockj = VectorBlock::sizeY * velInfo[i].index[1];
      const size_t blockStart = blocki + stride * blockj;

       velLab.load( velInfo[i], 0); const auto & __restrict__ V   = velLab;
      presLab.load(presInfo[i], 0); const auto & __restrict__ P   = presLab;
      uDefLab.load(uDefInfo[i], 0); const auto & __restrict__ UDEF= uDefLab;
      iRhoLab.load(iRhoInfo[i], 0); const auto & __restrict__ IRHO= iRhoLab;
      const auto& __restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            auto& __restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            auto& __restrict__ RHS = *(ScalarBlock*) pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const size_t idx = blockStart + ix + stride*iy;
        const Real divUx  =    V(ix+1,iy).u[0] -    V(ix,iy).u[0];
        const Real divVy  =    V(ix,iy+1).u[1] -    V(ix,iy).u[1];
        const Real UDEFW = (UDEF(ix  ,iy).u[0] + UDEF(ix-1,iy).u[0]) / 2;
        const Real UDEFE = (UDEF(ix+1,iy).u[0] + UDEF(ix  ,iy).u[0]) / 2;
        const Real VDEFS = (UDEF(ix,iy  ).u[1] + UDEF(ix,iy-1).u[1]) / 2;
        const Real VDEFN = (UDEF(ix,iy+1).u[1] + UDEF(ix,iy  ).u[1]) / 2;
        const Real divUS = UDEFE - UDEFW + VDEFN - VDEFS;
        TMP(ix, iy).s = facDiv*(divUx+divVy - CHI(ix,iy).s*divUS);

        const Real rE = (IRHO(ix+1,iy).s + IRHO(ix,iy).s)/2;
        const Real rW = (IRHO(ix-1,iy).s + IRHO(ix,iy).s)/2;
        const Real rN = (IRHO(ix,iy+1).s + IRHO(ix,iy).s)/2;
        const Real rS = (IRHO(ix,iy-1).s + IRHO(ix,iy).s)/2;
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        RHS(ix,iy).s = TMP(ix,iy).s +(1-rE)*dE -(1-rW)*dW +(1-rN)*dN -(1-rS)*dS;
        mat[idx][0] = - rN - rS - rE - rW;
        mat[idx][1] = rW; mat[idx][2] = rE; mat[idx][3] = rS; mat[idx][4] = rN;
      }


      for(int iy=0; iy<VectorBlock::sizeY && isE(velInfo[i]); ++iy) {
        TMP(VectorBlock::sizeX-1, iy).s = 0;
        RHS(VectorBlock::sizeX-1, iy).s = 0;
      }
      for(int ix=0; ix<VectorBlock::sizeX && isN(velInfo[i]); ++ix) {
        TMP(ix, VectorBlock::sizeY-1).s = 0;
        RHS(ix, VectorBlock::sizeY-1).s = 0;
      }
    }
  }
}

void PressureVarRho_proper::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 1, 1, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const auto &__restrict__ P = presLab;
      iRhoLab.load(iRhoInfo[i],0); const auto &__restrict__ IRHO = iRhoLab;
      auto& __restrict__ V = *(VectorBlock*) velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        // update vel field after most recent force and pressure response:
        const Real IRHOX = (IRHO(ix,iy).s + IRHO(ix-1,iy).s)/2;
        const Real IRHOY = (IRHO(ix,iy).s + IRHO(ix,iy-1).s)/2;
        V(ix,iy).u[0] += pFac * IRHOX * (P(ix,iy).s - P(ix-1,iy).s);
        V(ix,iy).u[1] += pFac * IRHOY * (P(ix,iy).s - P(ix,iy-1).s);
      }
    }
  }
}

void PressureVarRho_proper::operator()(const double dt)
{
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& tmpInfo  = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& rhsInfo  = sim.pRHS->getBlocksInfo();

  if(sim.step < 20) {
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
      auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
      auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) Pold(ix,iy).s = Pcur(ix,iy).s;
    }
  }

  sim.startProfiler("Prhs");
  updatePressureRHS(dt);
  fadeoutBorder(dt);
  sim.stopProfiler();

  if(sim.step < 20) {
    const Real fac = 1 - sim.step / 20.0;
    unifRhoSolver->solve(rhsInfo, presInfo);
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
      auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
      auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        Pcur(ix,iy).s += fac * (Pold(ix,iy).s - Pcur(ix,iy).s);
    }
  }

  #ifdef HYPREFFT
    varRhoSolver->solve(tmpInfo, presInfo);
    //pressureSolver->bUpdateMat = false;
  #else
    printf("Class PressureVarRho_proper REQUIRES HYPRE\n");
    fflush(0); abort();
  #endif

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureVarRho_proper::PressureVarRho_proper(SimulationData& s) : Operator(s),
#ifdef HYPREFFT
  varRhoSolver( new HYPREdirichletVarRho(s) ),
#endif
  unifRhoSolver( PoissonSolver::makeSolver(s) ) { }

PressureVarRho_proper::~PressureVarRho_proper() {
  #ifdef HYPREFFT
    delete varRhoSolver;
  #endif
    delete unifRhoSolver;
}

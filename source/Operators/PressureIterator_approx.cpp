//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureIterator_approx.h"
#include "../Poisson/HYPREdirichletVarRho.h"
#include "../Shape.h"
#include "Utils/BufferedLogger.h"
#include <stdio.h>

using namespace cubism;
// #define DECOUPLE
// #define EXPL_INTEGRATE_MOM

template<typename T>
static inline T mean(const T A, const T B) { return 0.5*(A+B); }

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PressureVarRho_approx::fadeoutBorder(const double dt) const
{
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const auto& extent = sim.extents;
  //const Real fadeLenX = sim.fadeLenX, fadeLenY = sim.fadeLenY;
  //const Real fadeLenX = 8*sim.getH(), fadeLenY = 8*sim.getH();
  const Real fadeLenX = extent[0]*0.05, fadeLenY = extent[1]*0.05;
  const Real invFadeX = 1/(fadeLenX+EPS), invFadeY = 1/(fadeLenY+EPS);
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
}

void PressureVarRho_approx::pressureCorrectionInit(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h;
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& vPresInfo= sim.vFluid->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);
    VectorLab velLab;   velLab.prepare(*(sim.vel),  stenBeg, stenEnd, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
       velLab.load( velInfo[i],0); const VectorLab& __restrict__ V   =  velLab;
      uDefLab.load(uDefInfo[i],0); const VectorLab& __restrict__ UDEF= uDefLab;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;
      const ScalarBlock&__restrict__  CHI= *(ScalarBlock*)  chiInfo[i].ptrBlock;
      // returns : pressure-corrected velocity and initial pressure eq RHS
      VectorBlock & __restrict__ vPres =  *(VectorBlock*) vPresInfo[i].ptrBlock;
      ScalarBlock & __restrict__ pRhs  =  *(ScalarBlock*)  pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
          const Real PE = P(ix+1,iy).s, PW = P(ix-1,iy).s;
          const Real PN = P(ix,iy+1).s, PS = P(ix,iy-1).s;
          const Real divVx  =    V(ix+1,iy).u[0] -    V(ix-1,iy).u[0];
          const Real divVy  =    V(ix,iy+1).u[1] -    V(ix,iy-1).u[1];
          const Real divUSx = UDEF(ix+1,iy).u[0] - UDEF(ix-1,iy).u[0];
          const Real divUSy = UDEF(ix,iy+1).u[1] - UDEF(ix,iy-1).u[1];
          vPres(ix,iy).u[0] = V(ix,iy).u[0] + pFac * (PE-PW) * IRHO(ix,iy).s;
          vPres(ix,iy).u[1] = V(ix,iy).u[1] + pFac * (PN-PS) * IRHO(ix,iy).s;
          pRhs(ix,iy).s = divVx+divVy - CHI(ix,iy).s * (divUSx+divUSy);
      }
    }
  }
}

Real PressureVarRho_approx::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, iRho0 = 1 / rho0;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& vPresInfo= sim.vFluid->getBlocksInfo();

  Real DP = 0, NP = 0;
  #pragma omp parallel reduction(+ : DP, NP)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ Pcur =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ P    = presLab;

      VectorBlock & __restrict__ vPres =  *(VectorBlock*) vPresInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;
      const VectorBlock&__restrict__    V= *(VectorBlock*)  velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dUdiv = (P(ix+1,iy).s-P(ix-1,iy).s)*(IRHO(ix,iy).s-iRho0);
        const Real dVdiv = (P(ix,iy+1).s-P(ix,iy-1).s)*(IRHO(ix,iy).s-iRho0);
        const Real dUpre = (Pcur(ix+1,iy).s - Pcur(ix-1,iy).s) * iRho0;
        const Real dVpre = (Pcur(ix,iy+1).s - Pcur(ix,iy-1).s) * iRho0;
        vPres(ix,iy).u[0] = V(ix,iy).u[0] + pFac * ( dUpre + dUdiv );
        vPres(ix,iy).u[1] = V(ix,iy).u[1] + pFac * ( dVpre + dVdiv );

        DP += std::pow(Pcur(ix,iy).s - P(ix,iy).s, 2);
        NP += std::pow(Pcur(ix,iy).s, 2);
      }
    }
  }

  return std::sqrt(DP / std::max(EPS, NP));
}

void PressureVarRho_approx::integrateMomenta(Shape * const shape) const
{
  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const std::vector<BlockInfo>& vFluidInfo = sim.vFluid->getBlocksInfo();

  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
  const double hsq = std::pow(velInfo[0].h_gridpoint, 2);
  double PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel for schedule(dynamic,1) reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock&__restrict__ V = *(VectorBlock*) vFluidInfo[i].ptrBlock;

    if(OBLOCK[vFluidInfo[i].blockID] == nullptr) continue;
    const CHI_MAT & __restrict__ rho = OBLOCK[ vFluidInfo[i].blockID ]->rho;
    const CHI_MAT & __restrict__ chi = OBLOCK[ vFluidInfo[i].blockID ]->chi;
    #ifndef EXPL_INTEGRATE_MOM
    const Real lambdt = sim.lambda * sim.dt;
    const UDEFMAT & __restrict__ udef = OBLOCK[ vFluidInfo[i].blockID ]->udef;
    #endif

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      #ifdef EXPL_INTEGRATE_MOM
        const Real F = hsq * rho[iy][ix] * chi[iy][ix];
        const Real udiff[2] = { V(ix,iy).u[0], V(ix,iy).u[1] };
      #else
        const Real Xlamdt = chi[iy][ix] * lambdt;
        const Real F = hsq * rho[iy][ix] * Xlamdt / (1 + Xlamdt);
        const Real udiff[2] = {
          V(ix,iy).u[0] - udef[iy][ix][0], V(ix,iy).u[1] - udef[iy][ix][1]
        };
      #endif
      double p[2]; vFluidInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      PM += F;
      PJ += F * (p[0]*p[0] + p[1]*p[1]);
      PX += F * p[0];
      PY += F * p[1];
      UM += F * udiff[0];
      VM += F * udiff[1];
      AM += F * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }

  shape->fluidAngMom = AM;
  shape->fluidMomX = UM;
  shape->fluidMomY = VM;
  shape->penalDX = PX;
  shape->penalDY = PY;
  shape->penalM = PM;
  shape->penalJ = PJ;
}

Real PressureVarRho_approx::penalize(const double dt) const
{
  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  //const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& vFluidInfo = sim.vFluid->getBlocksInfo();
  #ifdef EXPL_INTEGRATE_MOM
    const Real lambda = 1 / sim.dt;
  #else
    const Real lambda = sim.lambda;
  #endif

  Real MX = 0, MY = 0, DMX = 0, DMY = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : MX, MY, DMX, DMY)
  for (size_t i=0; i < Nblocks; i++)
  for (Shape * const shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
    if (o == nullptr) continue;

    const Real u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
    const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

    const CHI_MAT & __restrict__ X = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
          VectorBlock& __restrict__   F = *(VectorBlock*)  tmpVInfo[i].ptrBlock;
    //    ScalarBlock& __restrict__ FAC = *(ScalarBlock*)  pOldInfo[i].ptrBlock;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)   chiInfo[i].ptrBlock;
    const VectorBlock& __restrict__  UF = *(VectorBlock*)vFluidInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(CHI(ix,iy).s > X[iy][ix]) continue;
      if(X[iy][ix] <= 0) continue; // no need to do anything

      Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
      const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
      #ifdef EXPL_INTEGRATE_MOM
        const Real penalFac = lambda * X[iy][ix];
      #else
        const Real penalFac = lambda * X[iy][ix] / (1 +lambda * X[iy][ix] * dt);
      #endif
      const Real oldF[2] = {F(ix,iy).u[0], F(ix,iy).u[1]};
      F(ix,iy).u[0] = penalFac * ( US - UF(ix,iy).u[0] );
      F(ix,iy).u[1] = penalFac * ( VS - UF(ix,iy).u[1] );
      //F(ix,iy).u[0] = 0.1*F(ix,iy).u[0] + 0.9*penalFac*(US-UF(ix,iy).u[0]);
      //F(ix,iy).u[1] = 0.1*F(ix,iy).u[1] + 0.9*penalFac*(VS-UF(ix,iy).u[1]);
      const Real newF[2] = { F(ix,iy).u[0], F(ix,iy).u[1] };
      MX += std::pow(newF[0], 2); DMX += std::pow(newF[0]-oldF[0], 2);
      MY += std::pow(newF[1], 2); DMY += std::pow(newF[1]-oldF[1], 2);
    }
  }
  // return L2 relative momentum
  return std::sqrt((DMX + DMY) / (EPS + MX + MY));
}

#if 0
void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*rho0*h/dt;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& penlInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();

  Real sumRHSF=0, sumABSF=0, sumRHSS=0, sumABSS=0, sumRHSP=0, sumABSP=0;
  #pragma omp parallel reduction(+ : sumRHSP, sumABSP)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    //ScalarLab facLab; facLab.prepare(*(sim.pOld),   stenBeg, stenEnd, 0);
    VectorLab penlLab; penlLab.prepare(*(sim.tmpV),   stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      penlLab.load(penlInfo[i],0); const VectorLab& __restrict__ F = penlLab;
      //facLab.load(pOldInfo[i],0); const ScalarLab& __restrict__ FAC = facLab;
      //VectorBlock& __restrict__    F = *(VectorBlock*) penlInfo[i].ptrBlock;
      ScalarBlock& __restrict__ RHSS = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        //const Real divFx = (FAC(ix+1,iy).s - FAC(ix-1,iy).s) * F(ix,iy).u[0];
        //const Real divFy = (FAC(ix,iy+1).s - FAC(ix,iy-1).s) * F(ix,iy).u[1];
        //F(ix,iy).u[0] *= FAC(ix,iy).s;
        //F(ix,iy).u[1] *= FAC(ix,iy).s;
        const Real divFx = F(ix+1,iy  ).u[0] - F(ix-1,iy  ).u[0];
        const Real divFy = F(ix  ,iy+1).u[1] - F(ix  ,iy-1).u[1];
        RHSS(ix,iy).s = facDiv*dt * (divFx + divFy);
        sumABSP += h*h*std::fabs(RHSS(ix,iy).s);
        sumRHSP += h*h*          RHSS(ix,iy).s;
      }
    }
  }
  const Real corrP = 1 * sumRHSP / std::max(EPS, sumABSP);

  #pragma omp parallel reduction(+ : sumRHSF, sumABSF, sumRHSS, sumABSS)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
      ScalarBlock& __restrict__ RHSS  =*(ScalarBlock*)  tmpInfo[i].ptrBlock;
      ScalarBlock& __restrict__ RHSF  =*(ScalarBlock*) pOldInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ vRHS =*(ScalarBlock*)pRhsInfo[i].ptrBlock;
      const ScalarBlock& __restrict__  CHI =*(ScalarBlock*) chiInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real rE = (1 -rho0 * mean(IRHO(ix+1,iy).s, IRHO(ix,iy).s));
        const Real rW = (1 -rho0 * mean(IRHO(ix-1,iy).s, IRHO(ix,iy).s));
        const Real rN = (1 -rho0 * mean(IRHO(ix,iy+1).s, IRHO(ix,iy).s));
        const Real rS = (1 -rho0 * mean(IRHO(ix,iy-1).s, IRHO(ix,iy).s));
        // gradP at midpoints, skip factor 1/h, but skip multiply by h^2 later
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        // hatPfac=div((1-1/rho^*) gradP), div gives missing 1/h to make h^2
        const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        const Real divVfac = facDiv * vRHS(ix,iy).s;
        RHSF(ix,iy).s =     (1-CHI(ix,iy).s) * (hatPfac + divVfac);
        const Real rhsSolid =  CHI(ix,iy).s  * (hatPfac + divVfac);
        const Real rhsPenal = RHSS(ix,iy).s  - std::fabs(RHSS(ix,iy).s) * corrP;
        RHSS(ix,iy).s = rhsPenal + rhsSolid;
        sumABSF += h*h*std::fabs(RHSF(ix,iy).s);
        sumRHSF += h*h*          RHSF(ix,iy).s;
        sumABSS += h*h*std::fabs(rhsSolid);
        sumRHSS += h*h*          rhsSolid;
      }
    }
  }

  const Real corrS = 0 * sumRHSS / std::max(EPS, sumABSS);
  const Real corrF = 0 * sumRHSF / std::max(EPS, sumABSF);
  //printf("Relative divF RHS correction:%e %e\n", corrS, corrF);
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    const ScalarBlock&__restrict__ RHSF =*(ScalarBlock*) pOldInfo[i].ptrBlock;
          ScalarBlock&__restrict__ RHS  =*(ScalarBlock*)  tmpInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      const Real rhsFluid = RHSF(ix,iy).s - std::fabs(RHSF(ix,iy).s) * corrF;
      const Real rhsSolid = RHS (ix,iy).s - std::fabs(RHS (ix,iy).s) * corrS;
      RHS(ix,iy).s = rhsFluid + rhsSolid;
    }
  }
}
#elif 1
void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*rho0*h/dt;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
      ScalarBlock& __restrict__ RHS  = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ vRHS =*(ScalarBlock*)pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real rE = (1 -rho0 * mean(IRHO(ix+1,iy).s, IRHO(ix,iy).s));
        const Real rW = (1 -rho0 * mean(IRHO(ix-1,iy).s, IRHO(ix,iy).s));
        const Real rN = (1 -rho0 * mean(IRHO(ix,iy+1).s, IRHO(ix,iy).s));
        const Real rS = (1 -rho0 * mean(IRHO(ix,iy-1).s, IRHO(ix,iy).s));
        // gradP at midpoints, skip factor 1/h, but skip multiply by h^2 later
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        // hatPfac=div((1-1/rho^*) gradP), div gives missing 1/h to make h^2
        const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        RHS(ix,iy).s = facDiv * vRHS(ix,iy).s + hatPfac;
      }
    }
  }
}
#else
void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*rho0*h/dt;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& penlInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);
    VectorLab penlLab; penlLab.prepare(*(sim.tmpV),   stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
      penlLab.load(penlInfo[i],0); const VectorLab& __restrict__ F   = penlLab;
      ScalarBlock& __restrict__ RHS = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ vRHS =*(ScalarBlock*)pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real rE = (1 -rho0 * mean(IRHO(ix+1,iy).s, IRHO(ix,iy).s));
        const Real rW = (1 -rho0 * mean(IRHO(ix-1,iy).s, IRHO(ix,iy).s));
        const Real rN = (1 -rho0 * mean(IRHO(ix,iy+1).s, IRHO(ix,iy).s));
        const Real rS = (1 -rho0 * mean(IRHO(ix,iy-1).s, IRHO(ix,iy).s));
        // gradP at midpoints, skip factor 1/h, but skip multiply by h^2 later
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        // hatPfac=div((1-1/rho^*) gradP), div gives missing 1/h to make h^2
        const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        const Real divFx = F(ix+1,iy  ).u[0] - F(ix-1,iy  ).u[0];
        const Real divFy = F(ix  ,iy+1).u[1] - F(ix  ,iy-1).u[1];
        const Real divVfac = facDiv*( vRHS(ix,iy).s + dt*(divFx + divFy) );
        RHS(ix,iy).s = hatPfac + divVfac;
      }
    }
  }
}
#endif

void PressureVarRho_approx::finalizePressure(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h, iRho0 = 1 / rho0;
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ Pcur =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ Pold = presLab;
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock& __restrict__  F = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        // update vel field after most recent force and pressure response:
        const Real dUpre = (Pcur(ix+1,iy).s-Pcur(ix-1,iy).s)*iRho0;
        const Real dVpre = (Pcur(ix,iy+1).s-Pcur(ix,iy-1).s)*iRho0;
        const Real dUdiv = (Pold(ix+1,iy).s-Pold(ix-1,iy).s)*(IRHO(ix,iy).s-iRho0);
        const Real dVdiv = (Pold(ix,iy+1).s-Pold(ix,iy-1).s)*(IRHO(ix,iy).s-iRho0);
        V(ix,iy).u[0] += dt * F(ix,iy).u[0] + pFac * (dUpre + dUdiv);
        V(ix,iy).u[1] += dt * F(ix,iy).u[1] + pFac * (dVpre + dVdiv);
      }
    }
  }
}

void PressureVarRho_approx::operator()(const double dt)
{
  // first copy velocity before either Pres or Penal onto tmpV
  const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();

  pressureCorrectionInit(dt);

  int iter = 0;
  Real relDF = 1e3, relDP = 1e3;
  bool bConverged = false;
  for(iter = 0; ; iter++)
  {
    sim.startProfiler("Obj_force");
    for(Shape * const shape : sim.shapes)
    {
      // integrate vel in velocity after PP
      integrateMomenta(shape);
      shape->updateVelocity(dt);
    }

     // finally update vel with penalization but without pressure
    relDF = penalize(dt);
    sim.stopProfiler();

    // pressure solver is going to use as RHS = div VEL - \chi div UDEF
    sim.startProfiler("Prhs");
    updatePressureRHS(dt);
    fadeoutBorder(dt);
    sim.stopProfiler();

    if(0) //sim.step % 100 == 0)
    {
      char fname[512]; sprintf(fname, "RHS_%06d_%03d", sim.step, iter);
      sim.dumpTmp2( std::string(fname) );
    }

    pressureSolver->solve(tmpInfo, tmpInfo);

    {
      sim.startProfiler("PCorrect");
      relDP = pressureCorrection(dt);
      sim.stopProfiler();
    }
    printf("iter:%02d - rel. err penal:%e press:%e\n", iter, relDF, relDP);
    bConverged = std::max(relDF,relDP)<targetRelError ||
                 iter>2*oldNsteps || iter>100;

    if(bConverged)
    {
      sim.startProfiler("PCorrect");
      finalizePressure(dt);
      sim.stopProfiler();
    }

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const auto& __restrict__ Pnew = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
            auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dPold = Pcur(ix,iy).s - Pold(ix,iy).s;
        const Real dPnew = Pnew(ix,iy).s - Pcur(ix,iy).s;
        Pold(ix,iy).s = Pcur(ix,iy).s;
        if(dPnew*dPold>0) Pcur(ix,iy).s = 1.1*Pnew(ix,iy).s - .1*Pcur(ix,iy).s;
        else              Pcur(ix,iy).s =  .9*Pnew(ix,iy).s + .1*Pcur(ix,iy).s;
      }
      //Pold.copy(Pcur); Pcur.copy(Pnew);
    }

    if(bConverged) break;
  }

  oldNsteps = iter+1;
  if(oldNsteps > 30) targetRelError = std::max({relDF,relDP,targetRelError});
  if(oldNsteps > 10) targetRelError *= 1.01;
  if(oldNsteps <= 2) targetRelError *= 0.99;
  targetRelError = std::min(1e-3, std::max(1e-5, targetRelError));
  if(not sim.muteAll)
  {
    std::stringstream ssF; ssF<<sim.path2file<<"/pressureIterStats.dat";
    std::ofstream pfile(ssF.str().c_str(), std::ofstream::app);
    if(sim.step==0) pfile<<"step time dt iter relDF"<<"\n";
    pfile<<sim.step<<" "<<sim.time<<" "<<sim.dt<<" "<<iter<<" "<<relDF<<"\n";
  }
}

PressureVarRho_approx::PressureVarRho_approx(SimulationData& s) : Operator(s), pressureSolver( PoissonSolver::makeSolver(s) )  { }

PressureVarRho_approx::~PressureVarRho_approx() {
    delete pressureSolver;
}

#if 0
Real PressureVarRho_approx::penalize(const double dt) const
{
  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& vFluidInfo = sim.vFluid->getBlocksInfo();
  #ifdef EXPL_INTEGRATE_MOM
    const Real lambda = 1 / sim.dt;
  #else
    const Real lambda = sim.lambda;
  #endif

  Real MX = 0, MY = 0, DMX = 0, DMY = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : MX, MY, DMX, DMY)
  for (size_t i=0; i < Nblocks; i++)
  for (Shape * const shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
    if (o == nullptr) continue;

    const Real u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
    const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

    const CHI_MAT & __restrict__ X = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
          VectorBlock& __restrict__   F = *(VectorBlock*)  tmpVInfo[i].ptrBlock;
          ScalarBlock& __restrict__ FAC = *(ScalarBlock*)  pOldInfo[i].ptrBlock;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)   chiInfo[i].ptrBlock;
    const VectorBlock& __restrict__  UF = *(VectorBlock*)vFluidInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(CHI(ix,iy).s > X[iy][ix]) continue;
      if(X[iy][ix] <= 0) continue; // no need to do anything

      Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
      const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
      #ifdef EXPL_INTEGRATE_MOM
        const Real penalFac = lambda * X[iy][ix];
      #else
        const Real penalFac = lambda * X[iy][ix] / (1 +lambda * X[iy][ix] * dt);
      #endif
      const Real oldF[2] = {F(ix,iy).u[0], F(ix,iy).u[1]};
      F(ix,iy).u[0] = US - UF(ix,iy).u[0];
      F(ix,iy).u[1] = VS - UF(ix,iy).u[1];
      FAC(ix,iy).s = penalFac;
      const Real newF[2] = {penalFac*F(ix,iy).u[0], penalFac*F(ix,iy).u[1]};
      MX += std::pow(newF[0],2); DMX += std::pow(newF[0]-oldF[0],2);
      MY += std::pow(newF[1],2); DMY += std::pow(newF[1]-oldF[1],2);
    }
  }
  // return L2 relative momentum
  return std::sqrt((DMX + DMY) / (EPS + MX + MY));
}

void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*rho0*h/dt;
  const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();

  Real sumRHSF = 0, sumABSF = 0, sumRHSS = 0, sumABSS = 0;
  #pragma omp parallel reduction(+ : sumRHSF, sumABSF, sumRHSS, sumABSS)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    //VectorLab penlLab; penlLab.prepare(*(sim.tmpV),   stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBeg, stenEnd, 0);
    ScalarLab  facLab;  facLab.prepare(*(sim.pOld),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      //penlLab.load(penlInfo[i],0); const VectorLab& __restrict__ F   = penlLab;
      VectorBlock& __restrict__   F = *(VectorBlock*)  tmpVInfo[i].ptrBlock;
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
       facLab.load(pOldInfo[i],0); const ScalarLab& __restrict__ FAC = facLab;

      ScalarBlock& __restrict__ RHSS  =*(ScalarBlock*)  tmpInfo[i].ptrBlock;
      ScalarBlock& __restrict__ RHSF  =*(ScalarBlock*) pOldInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ vRHS =*(ScalarBlock*)pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        RHSS(ix,iy).s = facDiv * dt * (
           (FAC(ix+1,iy).s-FAC(ix-1,iy).s) * F(ix,iy).u[0]
          +(FAC(ix,iy+1).s-FAC(ix,iy-1).s) * F(ix,iy).u[1] );
        F(ix,iy).u[0] *= FAC(ix,iy).s;
        F(ix,iy).u[1] *= FAC(ix,iy).s;
        sumABSS += std::fabs(RHSS(ix,iy).s);
        sumRHSS +=           RHSS(ix,iy).s;

        const Real rE = (1 -rho0 * mean(IRHO(ix+1,iy).s, IRHO(ix,iy).s));
        const Real rW = (1 -rho0 * mean(IRHO(ix-1,iy).s, IRHO(ix,iy).s));
        const Real rN = (1 -rho0 * mean(IRHO(ix,iy+1).s, IRHO(ix,iy).s));
        const Real rS = (1 -rho0 * mean(IRHO(ix,iy-1).s, IRHO(ix,iy).s));
        // gradP at midpoints, skip factor 1/h, but skip multiply by h^2 later
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        // hatPfac=div((1-1/rho^*) gradP), div gives missing 1/h to make h^2
        const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        RHSF(ix,iy).s = facDiv * vRHS(ix,iy).s + hatPfac;
        sumABSF += std::fabs(RHSF(ix,iy).s);
        sumRHSF +=           RHSF(ix,iy).s;
      }
    }
  }

  sumABSS = std::max(std::numeric_limits<Real>::epsilon(), sumABSS);
  sumABSF = std::max(std::numeric_limits<Real>::epsilon(), sumABSF);
  const Real corrS = sumRHSS / sumABSS, corrF = sumRHSF / sumABSF;
  //printf("Relative divF RHS correction:%e %e\n", corrS, corrF);
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    const ScalarBlock&__restrict__ RHSF =*(ScalarBlock*) pOldInfo[i].ptrBlock;
          ScalarBlock&__restrict__ RHS  =*(ScalarBlock*)  tmpInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      const Real rhsFluid = RHSF(ix,iy).s - std::fabs(RHSF(ix,iy).s) * corrF;
      const Real rhsSolid = RHS (ix,iy).s - std::fabs(RHS (ix,iy).s) * corrS;
      RHS(ix,iy).s = rhsFluid + rhsSolid;
    }
  }
}
#endif

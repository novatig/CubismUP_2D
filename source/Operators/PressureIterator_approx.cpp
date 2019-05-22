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
// #define FOURTHORDER
#define EXPL_INTEGRATE_MOM

template<typename T>
static inline T mean(const T A, const T B) { return (A+B)/2; }
//static inline T mean(const T A, const T B) { return 2*A*B/(A+B); }

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
static constexpr double EPS = std::numeric_limits<double>::epsilon();

#ifndef FOURTHORDER
  static constexpr int stenBeg [3] = {-1,-1, 0}, stenEnd [3] = { 2, 2, 1};
  //static constexpr int stenBegP[3] = {-1,-1, 0}, stenEndP[3] = { 2, 2, 1};
  static constexpr int stenBegP[3] = {-2,-2, 0}, stenEndP[3] = { 3, 3, 1};

  static inline Real finDiffX(const ScalarLab& B, const int ix, const int iy) {
    return ( B(ix+1,iy).s - B(ix-1,iy).s ) / 2;
  }
  static inline Real finDiffY(const ScalarLab& B, const int ix, const int iy) {
    return ( B(ix,iy+1).s - B(ix,iy-1).s ) / 2;
  }
  static inline Real laplacian(const ScalarLab&B, const int ix, const int iy) {
    //return B(ix+1,iy).s +B(ix-1,iy).s +B(ix,iy+1).s +B(ix,iy-1).s -4*B(ix,iy).s;
    return ( finDiffX(B,ix+1,iy)-finDiffX(B,ix-1,iy)
            +finDiffY(B,ix,iy+1)-finDiffY(B,ix,iy-1) ) / 2;
  }
  template<int i>
  static inline Real finDiffX(const VectorLab& B, const int ix, const int iy) {
    return (B(ix+1,iy).u[i] - B(ix-1,iy).u[i]) / 2;
  }
  template<int i>
  static inline Real finDiffY(const VectorLab& B, const int ix, const int iy) {
    return (B(ix,iy+1).u[i] - B(ix,iy-1).u[i]) / 2;
  }
#else
  static constexpr int stenBeg [3] = {-2,-2, 0}, stenEnd [3] = { 3, 3, 1};
  static constexpr int stenBegP[3] = {-4,-4, 0}, stenEndP[3] = { 5, 5, 1};

  static inline Real finDiffX(const ScalarLab& B, const int ix, const int iy) {
    return (-B(ix+2,iy).s +8*B(ix+1,iy).s -8*B(ix-1,iy).s +B(ix-2,iy).s) / 12;
  }
  static inline Real finDiffY(const ScalarLab& B, const int ix, const int iy) {
    return (-B(ix,iy+2).s +8*B(ix,iy+1).s -8*B(ix,iy-1).s +B(ix,iy-2).s) / 12;
  }
  static inline Real laplacian(const ScalarLab&B, const int ix, const int iy) {
    //return (-B(ix+2,iy).s + 16*B(ix+1,iy).s + 16*B(ix-1,iy).s - B(ix-2,iy).s
    //        -B(ix,iy+2).s + 16*B(ix,iy+1).s + 16*B(ix,iy-1).s - B(ix,iy-2).s
    //        -60*B(ix,iy).s ) / 12;
    return ( -  finDiffX(B,ix+2,iy) +8*finDiffX(B,ix+1,iy)
             -8*finDiffX(B,ix-1,iy) +  finDiffX(B,ix-2,iy)
             -  finDiffY(B,ix,iy+2) +8*finDiffY(B,ix,iy+1)
             -8*finDiffY(B,ix,iy-1) +  finDiffY(B,ix,iy-2) ) / 12;
  }
  template<int i>
  static inline Real finDiffX(const VectorLab& B, const int ix, const int iy) {
    return (-B(ix+2,iy).u[i] +8*B(ix+1,iy).u[i] -8*B(ix-1,iy).u[i] +B(ix-2,iy).u[i]) / 12;
  }
  template<int i>
  static inline Real finDiffY(const VectorLab& B, const int ix, const int iy) {
    return (-B(ix,iy+2).u[i] +8*B(ix,iy+1).u[i] -8*B(ix,iy-1).u[i] +B(ix,iy-2).u[i]) / 12;
  }
#endif

void PressureVarRho_approx::fadeoutBorder(const double dt) const
{
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const auto& extent = sim.extents;
  //const Real fadeLenX = sim.fadeLenX, fadeLenY = sim.fadeLenY;
  //const Real fadeLenX = 8*sim.getH(), fadeLenY = 8*sim.getH();
  const Real fadeLenX = extent[0]*0.05, fadeLenY = extent[1]*0.05;
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
  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>&vPresInfo = sim.vFluid->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  //const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
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
        const Real divVf  = finDiffX<0>(V   ,ix,iy) + finDiffY<1>(V   ,ix,iy);
        const Real divDef = finDiffX<0>(UDEF,ix,iy) + finDiffY<1>(UDEF,ix,iy);
        vPres(ix,iy).u[0] = V(ix,iy).u[0] +pFac*finDiffX(P,ix,iy)*IRHO(ix,iy).s;
        vPres(ix,iy).u[1] = V(ix,iy).u[1] +pFac*finDiffY(P,ix,iy)*IRHO(ix,iy).s;
        pRhs(ix,iy).s = divVf - CHI(ix,iy).s * divDef;
      }

      //((VectorBlock*) uDefInfo[i].ptrBlock)->clear();
      //((VectorBlock*) tmpVInfo[i].ptrBlock)->clear();
    }
  }
}

Real PressureVarRho_approx::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& vPresInfo= sim.vFluid->getBlocksInfo();

  Real DP = 0, NP = 0;
  #pragma omp parallel reduction(+ : DP, NP)
  {
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ Pcur =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ P    = presLab;
      VectorBlock & __restrict__    vPres= *(VectorBlock*)vPresInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;
      const VectorBlock&__restrict__    V= *(VectorBlock*)  velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        #if 0
          const Real dUdiv = finDiffX(Pcur,ix,iy) * IRHO(ix,iy).s, dUpre = 0;
          const Real dVdiv = finDiffY(Pcur,ix,iy) * IRHO(ix,iy).s, dVpre = 0;
        #else
          const Real dUdiv = finDiffX(P,ix,iy) * (IRHO(ix,iy).s-iRho0);
          const Real dVdiv = finDiffY(P,ix,iy) * (IRHO(ix,iy).s-iRho0);
          const Real dUpre = finDiffX(Pcur,ix,iy) * iRho0;
          const Real dVpre = finDiffY(Pcur,ix,iy) * iRho0;
        #endif
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
  //printf("%e %e %e %e %e %e %e\n",AM,UM,VM,PX,PY,PM,PJ);
  #ifdef EXPL_INTEGRATE_MOM
  shape->fluidAngMom = AM;
  shape->fluidMomX = UM;
  shape->fluidMomY = VM;
  shape->penalDX = 0;
  shape->penalDY = 0;
  shape->penalM = PM;
  shape->penalJ = PJ;
  #else
  shape->fluidAngMom = AM;
  shape->fluidMomX = UM;
  shape->fluidMomY = VM;
  shape->penalDX = PX;
  shape->penalDY = PY;
  shape->penalM = PM;
  shape->penalJ = PJ;
  #endif
}

Real PressureVarRho_approx::penalize(const double dt) const
{
  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  //const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
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
    //    VectorBlock& __restrict__ olF= *(VectorBlock*)  uDefInfo[i].ptrBlock;
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
        const Real pFac = lambda * X[iy][ix];
      #else
        const Real pFac = lambda * X[iy][ix] / (1 +lambda * X[iy][ix] * dt);
      #endif
      const Real oldF[2] = {F(ix,iy).u[0], F(ix,iy).u[1]};
      const Real newF[2] = {pFac*(US-UF(ix,iy).u[0]), pFac*(VS-UF(ix,iy).u[1])};

      #if 1
        F(ix,iy).u[0] = newF[0]; F(ix,iy).u[1] = newF[1];
      #else
        const Real olrF[2] = {olF(ix,iy).u[0], olF(ix,iy).u[1]};
        const Real dUold = oldF[0]-olrF[0], dUnew = newF[0]-oldF[0];
        const Real dVold = oldF[1]-olrF[1], dVnew = newF[1]-oldF[1];
        if(dUold*dUnew>=0) F(ix,iy).u[0] = 1.2*newF[0] - 0.2*oldF[0];
        else F(ix,iy).u[0] = 0.9*newF[0] - 0.1*oldF[0];
        if(dVold*dVnew>=0) F(ix,iy).u[1] = 1.2*newF[1] - 0.2*oldF[1];
        else F(ix,iy).u[1] = 0.9*newF[1] - 0.1*oldF[1];
        olF(ix,iy).u[0] = oldF[0];
        olF(ix,iy).u[1] = oldF[1];
      #endif

      MX += std::pow(newF[0], 2); DMX += std::pow(newF[0]-oldF[0], 2);
      MY += std::pow(newF[1], 2); DMY += std::pow(newF[1]-oldF[1], 2);
    }
  }
  // return L2 relative momentum
  return std::sqrt((DMX + DMY) / (EPS + MX + MY));
}

void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = rho0 * h/dt;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& penlInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBegP, stenEndP, 0);
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
        #if 0
          const Real rE = (1 - rho0 * (IRHO(ix+1,iy).s + IRHO(ix,iy).s)/2);
          const Real rW = (1 - rho0 * (IRHO(ix-1,iy).s + IRHO(ix,iy).s)/2);
          const Real rN = (1 - rho0 * (IRHO(ix,iy+1).s + IRHO(ix,iy).s)/2);
          const Real rS = (1 - rho0 * (IRHO(ix,iy-1).s + IRHO(ix,iy).s)/2);
          const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
          const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
          const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        #else
          const Real lapP = (1 - rho0 * IRHO(ix,iy).s) * laplacian(P,ix,iy);
          const Real dPdRx = finDiffX(P,ix,iy) * finDiffX(IRHO,ix,iy);
          const Real dPdRy = finDiffY(P,ix,iy) * finDiffY(IRHO,ix,iy);
          const Real hatPfac = lapP - rho0 * (dPdRx + dPdRy);
        #endif
        const Real divF = finDiffX<0>(F,ix,iy) + finDiffY<1>(F,ix,iy);
        RHS(ix,iy).s = hatPfac + facDiv*( vRHS(ix,iy).s + dt*divF );
      }
    }
  }
}

void PressureVarRho_approx::finalizePressure(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ Pcur =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ Pold = presLab;
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock&__restrict__   F = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        #if 0
          const Real dUdiv = finDiffX(Pcur,ix,iy) * IRHO(ix,iy).s, dUpre = 0;
          const Real dVdiv = finDiffY(Pcur,ix,iy) * IRHO(ix,iy).s, dVpre = 0;
        #else
          const Real dUpre = finDiffX(Pcur,ix,iy) * iRho0;
          const Real dVpre = finDiffY(Pcur,ix,iy) * iRho0;
          const Real dUdiv = finDiffX(Pold,ix,iy) * (IRHO(ix,iy).s-iRho0);
          const Real dVdiv = finDiffY(Pold,ix,iy) * (IRHO(ix,iy).s-iRho0);
        #endif
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
  Real relDF = 1e3, relDP = 1e3;//, oldErr = 1e3;
  bool bConverged = false;
  for(iter = 0; ; iter++)
  {
    sim.startProfiler("Obj_force");
    for(Shape * const shape : sim.shapes) {
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

    if(0)//sim.step % 1000 == 0)
    {
      char fname[512]; sprintf(fname, "RHS_%06d_%03d", sim.step, iter);
      sim.dumpTmp2( std::string(fname) );
    }

    pressureSolver->solve(tmpInfo, tmpInfo);

    if(0)//sim.step % 1000 == 0)
    {
      char fname[512]; sprintf(fname, "P_%06d_%03d", sim.step, iter);
      sim.dumpTmp2( std::string(fname) );
    }

    sim.startProfiler("PCorrect");
    relDP = pressureCorrection(dt);
    sim.stopProfiler();

    printf("iter:%02d - rel. err penal:%e press:%e vel={%e %e}\n",
      iter, relDF, relDP, sim.shapes[0]->u, sim.shapes[0]->v);
    const Real newErr = std::max(relDF, relDP);
    bConverged =               newErr<targetRelError;
    bConverged = bConverged || iter>2*oldNsteps;
    bConverged = bConverged || iter>100;
    //bConverged = bConverged || (oldErr-newErr)/newErr < 0.01;
    //oldErr = newErr;

    if(bConverged) {
      sim.startProfiler("PCorrect");
      finalizePressure(dt);
      sim.stopProfiler();
    }

    //size_t nExtra=0, nInter=0;
    #pragma omp parallel for schedule(static) //reduction(+ : nExtra, nInter)
    for (size_t i=0; i < Nblocks; i++)
    {
      const auto& __restrict__ Pnew = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
            auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      #if 0
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dPold = Pcur(ix,iy).s - Pold(ix,iy).s;
        const Real dPnew = Pnew(ix,iy).s - Pcur(ix,iy).s;
        Pold(ix,iy).s = Pcur(ix,iy).s;
        if(dPnew*dPold>0) {
          Pcur(ix,iy).s = 1.5*Pnew(ix,iy).s -.5*Pcur(ix,iy).s; //++nExtra;
        } else {
          Pcur(ix,iy).s =  .8*Pnew(ix,iy).s +.2*Pcur(ix,iy).s; //++nInter;
        }
      }
      #else
      Pold.copy(Pcur);
      Pcur.copy(Pnew);
      #endif
    }
    //printf("nInter:%lu nExtra:%lu\n",nInter,nExtra);
    if(bConverged) break;
  }

  if(0)
  {
    char fname[512]; sprintf(fname, "P_%06d", sim.step);
    sim.dumpTmp2( std::string(fname) );
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
#elif 0
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
#endif
#endif

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

using namespace cubism;
#define ETA   0
#define ALPHA 1
#define DECOUPLE
#define EXPL_INTEGRATE_MOM

template<typename T>
static inline T mean(const T A, const T B) { return 0.5*(A+B); }

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PressureVarRho_approx::fadeoutBorder(const double dt) const
{
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
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
}

void PressureVarRho_approx::pressureCorrectionInit(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& vPresInfo= sim.vFluid->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);
    VectorLab velLab;   velLab.prepare(*(sim.vel),  stenBeg, stenEnd, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
       velLab.load( velInfo[i],0); const VectorLab& __restrict__ V   =  velLab;
      uDefLab.load(uDefInfo[i],0); const VectorLab& __restrict__ UDEF= uDefLab;
      const ScalarBlock&__restrict__  CHI= *(ScalarBlock*)  chiInfo[i].ptrBlock;
      // returns : pressure-corrected velocity and initial pressure eq RHS
      VectorBlock & __restrict__ vPres =  *(VectorBlock*) vPresInfo[i].ptrBlock;
      ScalarBlock & __restrict__ pRhs  =  *(ScalarBlock*)  pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        const Real iRhoX = mean(IRHO(ix,iy).s, IRHO(ix-1,iy).s);
        const Real iRhoY = mean(IRHO(ix,iy).s, IRHO(ix,iy-1).s);
        const Real divVx  =    V(ix+1,iy).u[0] -    V(ix,iy).u[0];
        const Real divVy  =    V(ix,iy+1).u[1] -    V(ix,iy).u[1];
        const Real divUSx = UDEF(ix+1,iy).u[0] - UDEF(ix,iy).u[0];
        const Real divUSy = UDEF(ix,iy+1).u[1] - UDEF(ix,iy).u[1];
        vPres(ix,iy).u[0] = V(ix,iy).u[0] +pFac*(P(ix,iy).s-P(ix-1,iy).s)*iRhoX;
        vPres(ix,iy).u[1] = V(ix,iy).u[1] +pFac*(P(ix,iy).s-P(ix,iy-1).s)*iRhoY;
        pRhs(ix,iy).s = divVx+divVy - CHI(ix,iy).s * (divUSx+divUSy);
      }
    }
  }
}

Real PressureVarRho_approx::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h, invRho0 = 1 / rho0;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();

  const std::vector<BlockInfo>& vPresInfo= sim.vFluid->getBlocksInfo();
  Real DP = 0, NP = 0;
  #pragma omp parallel reduction(+ : DP, NP)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 1, 1, 1};
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ P    =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ Pold = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab&__restrict__ IRHO = iRhoLab;

      VectorBlock & __restrict__ vPres =  *(VectorBlock*) vPresInfo[i].ptrBlock;
      const VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real invRhoX = mean(IRHO(ix,iy).s, IRHO(ix-1,iy).s);
        const Real invRhoY = mean(IRHO(ix,iy).s, IRHO(ix,iy-1).s);
        const Real dUdiv = (Pold(ix,iy).s-Pold(ix-1,iy).s)*(invRhoX - invRho0);
        const Real dVdiv = (Pold(ix,iy).s-Pold(ix,iy-1).s)*(invRhoY - invRho0);
        const Real dUpre = ( P(ix,iy).s - P(ix-1,iy).s )  * invRho0;
        const Real dVpre = ( P(ix,iy).s - P(ix,iy-1).s )  * invRho0;
        vPres(ix,iy).u[0] = V(ix,iy).u[0] + pFac * ( dUpre + dUdiv );
        vPres(ix,iy).u[1] = V(ix,iy).u[1] + pFac * ( dVpre + dVdiv );
        DP += std::pow(Pold(ix,iy).s - P(ix,iy).s, 2);
        NP += std::pow(P(ix,iy).s, 2);
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

  #pragma omp parallel reduction(+ : PM, PJ, PX, PY, UM, VM, AM)
  {
    static constexpr int stenBeg[3] = {0,0,0}, stenEnd[3] = {2,2,1};
    VectorLab velLab; velLab.prepare(*(sim.vFluid), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic,1)
    for(size_t i=0; i<Nblocks; i++)
    {
      if(OBLOCK[vFluidInfo[i].blockID] == nullptr) continue;

      velLab.load(vFluidInfo[i],0); const VectorLab& __restrict__ V = velLab;
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
        const Real UCC = (V(ix,iy).u[0] + V(ix+1,iy).u[0])/2;
        const Real VCC = (V(ix,iy).u[1] + V(ix,iy+1).u[1])/2;
        #ifdef EXPL_INTEGRATE_MOM
          const Real F = rho[iy][ix] * hsq * chi[iy][ix];
          const Real udiff[2] = { UCC, VCC };
        #else
          const Real XTL = chi[iy][ix]*lambdt, F = rho[iy][ix]*hsq*XTL/(1+XTL);
          const Real udiff[2] = { UCC-udef[iy][ix][0], VCC-udef[iy][ix][1] };
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
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& vFluidInfo = sim.vFluid->getBlocksInfo();
  #ifdef EXPL_INTEGRATE_MOM
    const Real lambda = 1 / sim.dt;
  #else
    const Real lambda = sim.lambda;
  #endif

  Real MX = 0, MY = 0, DMX = 0, DMY = 0;
  #pragma omp parallel reduction(+ : MX, MY, DMX, DMY)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab chiLab; chiLab.prepare(*(sim.chi), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic, 1)
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
      VectorBlock& __restrict__   F = *(VectorBlock*)   tmpVInfo[i].ptrBlock;
      VectorBlock& __restrict__  UF = *(VectorBlock*) vFluidInfo[i].ptrBlock;
      chiLab.load(chiInfo[i],0);  const ScalarLab& __restrict__ CHI = chiLab;

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
        const Real chiX = (CHI(ix,iy).s + CHI(ix-1,iy).s)/2;
        const Real chiY = (CHI(ix,iy).s + CHI(ix,iy-1).s)/2;
        #ifdef EXPL_INTEGRATE_MOM
          const Real penFacX = lambda * chiX, penFacY = lambda * chiY;
        #else
          const Real penFacX = lambda * chiX / (1 + lambda * chiX * dt);
          const Real penFacY = lambda * chiY / (1 + lambda * chiY * dt);
        #endif
        const Real oldF[2] = {F(ix,iy).u[0], F(ix,iy).u[1]};
        F(ix,iy).u[0] = penFacX * (US - UF(ix,iy).u[0]);
        F(ix,iy).u[1] = penFacY * (VS - UF(ix,iy).u[1]);
        //const Real uNext  = UF(ix,iy).u[0] + dt * F(ix,iy).u[0];
        //const Real vNext  = UF(ix,iy).u[1] + dt * F(ix,iy).u[1];
        // uPres now becomes delta Vel
        //UF(ix,iy).u[0] = dt * (uNext-US); UF(ix,iy).u[1] = dt * (vNext-VS);
        MX+= std::pow(F(ix,iy).u[0],2); DMX+= std::pow(F(ix,iy).u[0]-oldF[0],2);
        MY+= std::pow(F(ix,iy).u[1],2); DMY+= std::pow(F(ix,iy).u[1]-oldF[1],2);
      }
    }
  }
  // return L2 relative momentum
  return std::sqrt((DMX + DMY) / (EPS + MX + MY));
}

void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = rho0*h/dt;
  const std::vector<BlockInfo>& penlInfo = sim.tmpV->getBlocksInfo();
  //const std::vector<BlockInfo>&   dUInfo = sim.vFluid->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  //const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab penlLab; penlLab.prepare(*(sim.tmpV),   stenBeg, stenEnd, 0);
    //ScalarLab  chiLab;  chiLab.prepare(*(sim.chi),    stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
      penlLab.load(penlInfo[i],0); const VectorLab& __restrict__ PENL= penlLab;
      //chiLab.load( chiInfo[i],0); const ScalarLab& __restrict__   X =  chiLab;
      //const VectorBlock&__restrict__ deltaU=*(VectorBlock*)dUInfo[i].ptrBlock;
            ScalarBlock& __restrict__ TMP  =*(ScalarBlock*) tmpInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ pRhs =*(ScalarBlock*)pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real rE = (1 - rho0 * mean(IRHO(ix+1,iy).s, IRHO(ix,iy).s));
        const Real rW = (1 - rho0 * mean(IRHO(ix-1,iy).s, IRHO(ix,iy).s));
        const Real rN = (1 - rho0 * mean(IRHO(ix,iy+1).s, IRHO(ix,iy).s));
        const Real rS = (1 - rho0 * mean(IRHO(ix,iy-1).s, IRHO(ix,iy).s));
        // gradP at midpoints, skip factor 1/h, but skip multiply by h^2 later
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        // hatPfac=div((1-1/rho^*) gradP), div gives missing 1/h to make h^2
        const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        const Real divFx = PENL(ix+1,iy).u[0] - PENL(ix,iy).u[0];
        const Real divFy = PENL(ix,iy+1).u[1] - PENL(ix,iy).u[1];
        //const Real dUgradXx = (X(ix+1,iy).s-X(ix-1,iy).s) * deltaU(ix,iy).u[0];
        //const Real dUgradXy = (X(ix,iy+1).s-X(ix,iy-1).s) * deltaU(ix,iy).u[1];
        // +dUgradXx+dUgradXy
        const Real rhsDiv = pRhs(ix,iy).s + dt*(divFx+divFy);
        TMP(ix,iy).s = facDiv*rhsDiv + hatPfac;
      }
    }
  }
}

void PressureVarRho_approx::finalizePressure(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h, invRho0 = 1 / rho0;
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = {1,1,1};
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ P    =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ Pold = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab&__restrict__ IRHO = iRhoLab;
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock& __restrict__  F = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real invRhoX = mean(IRHO(ix,iy).s, IRHO(ix-1,iy).s);
        const Real invRhoY = mean(IRHO(ix,iy).s, IRHO(ix,iy-1).s);
        const Real dUdiv = (Pold(ix,iy).s-Pold(ix-1,iy).s)*(invRhoX - invRho0);
        const Real dVdiv = (Pold(ix,iy).s-Pold(ix,iy-1).s)*(invRhoY - invRho0);
        const Real dUpre = ( P(ix,iy).s - P(ix-1,iy).s ) * invRho0;
        const Real dVpre = ( P(ix,iy).s - P(ix,iy-1).s ) * invRho0;
        V(ix,iy).u[0] += dt*F(ix,iy).u[0] + pFac * ( dUpre + dUdiv );
        V(ix,iy).u[1] += dt*F(ix,iy).u[1] + pFac * ( dVpre + dVdiv );
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
  for(iter = 0; iter < 1000; iter++)
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

    pressureSolver->solve(tmpInfo, tmpInfo);

    if(bConverged)
    {
      sim.startProfiler("PCorrect");
      finalizePressure(dt);
      sim.stopProfiler();
    }
    else
    {
      sim.startProfiler("PCorrect");
      relDP = pressureCorrection(dt);
      sim.stopProfiler();
    }
    printf("iter:%02d - rel. err penal:%e press:%e\n", iter, relDF, relDP);
    relDF = std::max(relDF, relDP);

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const auto& __restrict__ Pnew = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
            auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      Pold.copy(Pcur);
      Pcur.copy(Pnew);
    }

    if(bConverged) break;
    bConverged = relDF<targetRelError || iter>2*oldNsteps;
    // bConverged = true;
    // if penalization force converged, do one more Poisson solve
  }

  oldNsteps = iter+1;
  if(oldNsteps > 10) targetRelError *= 1.01;
  if(oldNsteps <= 2) targetRelError *= 0.99;

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

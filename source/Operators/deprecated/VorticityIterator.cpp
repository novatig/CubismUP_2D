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

void VorticityIterator::fadeoutBorder(const double dt) const
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

void VorticityIterator::pressureCorrectionInit(const double dt) const
{
}

Real VorticityIterator::computeVelocity() const
{
  const Real h = sim.getH(), invH = 1/h;
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();

  #pragma omp parallel
  {
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ PSI =  tmpLab;
      VectorBlock&__restrict__ V = *(VectorBlock*) velInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        V(ix,iy).u[0] = + invH * finDiffY(P,ix,iy);
        V(ix,iy).u[1] = - invH * finDiffX(P,ix,iy);
      }
    }
  }
}

void VorticityIterator::integrateMomenta(Shape * const shape) const
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

Real VorticityIterator::penalize(const double dt) const
{
  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& omegaInfo = sim.omega->getBlocksInfo();

  #ifdef EXPL_INTEGRATE_MOM
    const Real lambda = 1 / sim.dt;
  #else
    const Real lambda = sim.lambda;
  #endif

  Real MX = 0, MY = 0, DMX = 0, DMY = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : MX, MY, DMX, DMY)
  for (size_t i=0; i < Nblocks; i++)
  {
  VectorBlock& __restrict__ F = *(VectorBlock*)  tmpVInfo[i].ptrBlock;
  F.clear();

  for (Shape * const shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
    if (o == nullptr) continue;

    const Real u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
    const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
    const CHI_MAT & __restrict__ X = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[i].ptrBlock;
    const VectorBlock& __restrict__   V = *(VectorBlock*) velInfo[i].ptrBlock;

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

      F(ix,iy).u[0] = newF[0]; V(ix,iy).u[0] += dt * newF[0];
      F(ix,iy).u[1] = newF[1]; V(ix,iy).u[1] += dt * newF[1];

      MX += std::pow(newF[0], 2); DMX += std::pow(newF[0]-oldF[0], 2);
      MY += std::pow(newF[1], 2); DMY += std::pow(newF[1]-oldF[1], 2);
    }
  }
  }
  // return L2 relative momentum
  return std::sqrt((DMX + DMY) / (EPS + MX + MY));
}

void VorticityIterator::vorticityRHS() const
{
  const Real h = sim.getH();
  const std::vector<BlockInfo>& omegaInfo = sim.omega->getBlocksInfo();
  const std::vector<BlockInfo>&   tmpInfo = sim.tmp->getBlocksInfo();
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++) {
          ScalarBlock&__restrict__ RHS = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
    const ScalarBlock&__restrict__ OMG = *(ScalarBlock*)omegaInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) RHS(ix,iy).s = -h*h*OMG(ix,iy).s;
  }
}

void VorticityIterator::finalizeVorticity(const double dt) const
{
  const Real h = sim.getH(), fFac = dt/h;
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& omegaInfo = sim.omega->getBlocksInfo();

  #pragma omp parallel
  {
    VectorLab tmpLab;  tmpLab.prepare(*(sim.tmpV), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      tmpLab.load(tmpVInfo[i],0); const VectorLab&__restrict__ F =  tmpLab;
      ScalarBlock& __restrict__ OMG = *(ScalarBlock*) omegaInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        OMG(ix,iy).s += fFac * ( finDiffX<1>(F,ix,iy) - finDiffY<0>(F,ix,iy) );
    }
  }
}

void VorticityIterator::operator()(const double dt)
{
  int iter = 0;
  Real relDF = 1e3, relDP = 1e3;//, oldErr = 1e3;
  bool bConverged = false;

  for(iter = 0; ; iter++)
  {
    sim.startProfiler("rhs");
    vorticityRHS();
    fadeoutBorder(dt);
    sim.stopProfiler();

    pressureSolver->solve(tmpInfo, tmpInfo);

    sim.startProfiler("Obj_force");
    for(Shape * const shape : sim.shapes) {
      // integrate vel in velocity after PP
      integrateMomenta(shape);
      shape->updateVelocity(dt);
    }

     // finally update vel with penalization but without pressure
    relDF = penalize(dt);
    sim.stopProfiler();

    sim.startProfiler("Correct");
    relDP = finalizeVorticity(dt);
    sim.stopProfiler();

    printf("iter:%02d - rel. err penal:%e press:%e vel={%e %e}\n",
      iter, relDF, relDP, sim.shapes[0]->u, sim.shapes[0]->v);
    const Real newErr = std::max(relDF, relDP);
    bConverged = true; //              newErr<targetRelError;
    bConverged = bConverged || iter>2*oldNsteps;
    bConverged = bConverged || iter>100;

    //printf("nInter:%lu nExtra:%lu\n",nInter,nExtra);
    if(bConverged) break;
  }

  oldNsteps = iter+1;
  if(oldNsteps > 30) targetRelError = std::max({relDF,relDP,targetRelError});
  if(oldNsteps > 10) targetRelError *= 1.01;
  if(oldNsteps <= 2) targetRelError *= 0.99;
  targetRelError = std::min(1e-3, std::max(1e-5, targetRelError));

}

VorticityIterator::VorticityIterator(SimulationData& s) : Operator(s), pressureSolver( PoissonSolver::makeSolver(s) )  { }

VorticityIterator::~VorticityIterator() {
    delete pressureSolver;
}

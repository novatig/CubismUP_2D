//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "UpdateObjects.h"
#include "../Shape.h"

using namespace cubism;

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

//#define EXPL_INTEGRATE_MOM

void UpdateObjects::preventCollidingObstacles() const
{
  const std::vector<Shape*>& shapes = sim.shapes;
  const size_t N = shapes.size();

  struct CollisionInfo // hitter and hittee, symmetry but we do things twice
  {
    Real iM = 0, iPosX = 0, iPosY = 0, iMomX = 0, iMomY = 0;
    Real jM = 0, jPosX = 0, jPosY = 0, jMomX = 0, jMomY = 0;
  };
  std::vector<CollisionInfo> collisions(N);

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i<N; ++i)
  for (size_t j=0; j<N; ++j)
  {
    if(i==j) continue;
    auto & coll = collisions[i];
    const auto& iBlocks = shapes[i]->obstacleBlocks;
    const auto& jBlocks = shapes[j]->obstacleBlocks;
    const Real iUl = shapes[i]->u, iVl = shapes[i]->v, iW = shapes[i]->omega;
    const Real jUl = shapes[j]->u, jVl = shapes[j]->v, jW = shapes[j]->omega;
    const Real iCx =shapes[i]->centerOfMass[0], iCy =shapes[i]->centerOfMass[1];
    const Real jCx =shapes[j]->centerOfMass[0], jCy =shapes[j]->centerOfMass[1];

    assert(iBlocks.size() == jBlocks.size());
    const size_t nBlocks = iBlocks.size();

    for (size_t k=0; k<nBlocks; ++k)
    {
      if ( iBlocks[k] == nullptr || jBlocks[k] == nullptr ) continue;

      const CHI_MAT & iChi  = iBlocks[k]->chi,  & jChi  = jBlocks[k]->chi;
      const UDEFMAT & iUDEF = iBlocks[k]->udef, & jUDEF = jBlocks[k]->udef;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if(iChi[iy][ix] <= 0 || jChi[iy][ix] <= 0 ) continue;

        const auto pos = velInfo[k].pos<Real>(ix, iy);
        const Real iUr = - iW * (pos[1] - iCy), iVr =   iW * (pos[0] - iCx);
        const Real jUr = - jW * (pos[1] - jCy), jVr =   jW * (pos[0] - jCx);
        coll.iM    += iChi[iy][ix];
        coll.iPosX += iChi[iy][ix] * pos[0];
        coll.iPosY += iChi[iy][ix] * pos[1];
        coll.iMomX += iChi[iy][ix] * (iUl + iUr + iUDEF[iy][ix][0]);
        coll.iMomY += iChi[iy][ix] * (iVl + iVr + iUDEF[iy][ix][1]);
        coll.jM    += jChi[iy][ix];
        coll.jPosX += jChi[iy][ix] * pos[0];
        coll.jPosY += jChi[iy][ix] * pos[1];
        coll.jMomX += jChi[iy][ix] * (jUl + jUr + jUDEF[iy][ix][0]);
        coll.jMomY += jChi[iy][ix] * (jVl + jVr + jUDEF[iy][ix][1]);
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i<N; ++i)
  for (size_t j=0; j<N; ++j)
  {
    if(i==j) continue;
    auto & coll = collisions[i];

    // less than one fluid element of overlap: wait to get closer. no hit
    if(coll.iM < 1 || coll.jM < 1) continue;
    const Real iPX = coll.iPosX / coll.iM, iPY = coll.iPosY / coll.iM;
    const Real iUX = coll.iMomX / coll.iM, iUY = coll.iMomY / coll.iM;
    const Real jPX = coll.jPosX / coll.jM, jPY = coll.jPosY / coll.jM;
    const Real jUX = coll.jMomX / coll.jM, jUY = coll.jMomY / coll.jM;
    const Real CX = (iPX+jPX)/2, CY = (iPY+jPY)/2;
    const Real dirX = iPX - jPX, dirY = iPY - jPY;
    const Real hitVelX = jUX - iUX, hitVelY = jUY - iUY;
    const Real normF = std::max(std::sqrt(dirX*dirX + dirY*dirY), EPS);
    const Real NX = dirX / normF, NY = dirY / normF; // collision normal
    const Real projVel = hitVelX * NX + hitVelY * NY;
    printf("%lu hit %lu in [%f %f] with dir:[%f %f] DU:[%f %f] proj:%f\n",
        i, j, CX, CY, NX, NY, hitVelX, hitVelY, projVel); fflush(0);
    if(projVel<=0) continue; // vel goes away from coll: no need to bounce
    const bool iForcedX = shapes[i]->bForcedx && sim.time<shapes[i]->timeForced;
    const bool iForcedY = shapes[i]->bForcedy && sim.time<shapes[i]->timeForced;
    const bool iForcedA = shapes[i]->bBlockang&& sim.time<shapes[i]->timeForced;
    const bool jForcedX = shapes[j]->bForcedx && sim.time<shapes[j]->timeForced;
    const bool jForcedY = shapes[j]->bForcedy && sim.time<shapes[j]->timeForced;

    const Real iInvMassX = iForcedX? 0 : 1/shapes[i]->M; // forced == inf mass
    const Real iInvMassY = iForcedY? 0 : 1/shapes[i]->M;
    const Real jInvMassX = jForcedX? 0 : 1/shapes[j]->M;
    const Real jInvMassY = jForcedY? 0 : 1/shapes[j]->M;
    const Real iInvJ     = iForcedA? 0 : 1/shapes[i]->J;
    const Real meanMassX = 2 / std::max(iInvMassX + jInvMassX, EPS);
    const Real meanMassY = 2 / std::max(iInvMassY + jInvMassY, EPS);
    // Force_i_bounce _from_j = HARMONIC_MEAN_MASS * (Uj - Ui) / dt
    const Real FXdt = meanMassX * projVel * NX;
    const Real FYdt = meanMassY * projVel * NY;
    printf("%f %f %f %f %f\n", FXdt, FYdt, iInvMassX, iInvMassY, iInvJ); fflush(0);
    shapes[i]->u += FXdt * iInvMassX; // if forced, no update
    shapes[i]->v += FYdt * iInvMassY;
    const Real iCx =shapes[i]->centerOfMass[0], iCy =shapes[i]->centerOfMass[1];
    const Real RcrossF = (CX-iCx) * FYdt - (CY-iCy) * FXdt;
    shapes[i]->omega += iInvJ * RcrossF;
  }
}

void UpdateObjects::integrateMomenta(Shape * const shape) const
{
  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
  const double hsq = std::pow(velInfo[0].h_gridpoint, 2);
  double PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel for schedule(dynamic,1) reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;

    if(OBLOCK[velInfo[i].blockID] == nullptr) continue;
    const CHI_MAT & __restrict__ rho = OBLOCK[velInfo[i].blockID]->rho;
    const CHI_MAT & __restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
    const UDEFMAT & __restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
    #ifndef EXPL_INTEGRATE_MOM
      const Real lambdt = sim.lambda * sim.dt;
    #endif

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      const Real udiff[2] = {
        VEL(ix,iy).u[0] - udef[iy][ix][0], VEL(ix,iy).u[1] - udef[iy][ix][1]
      };
      #ifdef EXPL_INTEGRATE_MOM
        const Real F = hsq * rho[iy][ix] * chi[iy][ix];
      #else
        const Real Xlamdt = chi[iy][ix] * lambdt;
        const Real F = hsq * rho[iy][ix] * Xlamdt / (1 + Xlamdt);
      #endif
      double p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      PM += F;
      PJ += F * (p[0]*p[0] + p[1]*p[1]);
      PX += F * p[0];  PY += F * p[1];
      UM += F * udiff[0]; VM += F * udiff[1];
      AM += F * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }

  shape->fluidAngMom = AM; shape->fluidMomX = UM; shape->fluidMomY = VM;
  shape->penalDX=PX; shape->penalDY=PY; shape->penalM=PM; shape->penalJ=PJ;
}

void UpdateObjects::penalize(const double dt) const
{
  #pragma omp parallel for schedule(dynamic, 1)
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
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)chiInfo[i].ptrBlock;
          VectorBlock& __restrict__   V = *(VectorBlock*)velInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(CHI(ix,iy).s > X[iy][ix]) continue;
      if(X[iy][ix] <= 0) continue; // no need to do anything

      Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      #ifndef EXPL_INTEGRATE_MOM
        const Real alpha = 1/(1 + sim.lambda * dt * X[iy][ix]);
      #else
        const Real alpha = 1 - X[iy][ix];
      #endif

      const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
      const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
      V(ix,iy).u[0] = alpha*V(ix,iy).u[0] + (1-alpha)*US;
      V(ix,iy).u[1] = alpha*V(ix,iy).u[1] + (1-alpha)*VS;
    }
  }
}

void UpdateObjects::operator()(const double dt)
{
  // penalization force is now assumed to be finalized
  // 1) integrate momentum
  sim.startProfiler("Obj_force");
  for(Shape * const shape : sim.shapes) {
    integrateMomenta(shape);
    shape->updateVelocity(dt);
  }
  preventCollidingObstacles();
  penalize(dt);
  sim.stopProfiler();
}

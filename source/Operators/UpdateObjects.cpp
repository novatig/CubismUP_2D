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
  const Real h = velInfo[k].h_gridpoint, invh = 1.0/h;
  const std::vector<Shape*>& shapes = sim.shapes;
  const size_t n = shapes.size();

  // iterate over surface of the hittee to have normal of the 'wall'
  // return normal and location of hit:
  const auto findIntersect = [&] (const ObstacleBlock * const oHitter,
                                  const ObstacleBlock * const oHittee,
                                  const BlockInfo & info )
  {
    const CHI_MAT & chiEr  = oHitter->chi;
    const auto& surfEe = oHittee->surface;
    const int nSurfEe = surfEe.size();

    Real intPx = 0, intPy = 0, intNorm = 0, intDchiX = 0, intDchiY = 0;

    for(int k = 0; k < nSurfEe; ++k)
    {
      const int ix = surfEe[k]->ix, iy = surfEe[k]->iy;
      if(chiEr[iy][ix] <= 0) continue;
      const std::array<Real,2> pos = info.pos<Real>(ix, iy);
      intDchiX += chiEr[iy][ix] * surfEe[k]->dchidx;
      intDchiY += chiEr[iy][ix] * surfEe[k]->dchidy;
      intPx    += chiEr[iy][ix] * surfEe[k]->delta * pos[0];
      intPy    += chiEr[iy][ix] * surfEe[k]->delta * pos[1];
      intNorm  += chiEr[iy][ix] * surfEe[k]->delta;
    }
    if(intNorm <= 0) return std::vector<Real> (0); // no hit
    const Real normF = std::sqrt(intDchiX * intDchiX + intDchiY * intDchiY);
    assert(normF>0 && "FATAL: if delta>0 also gradchi must be nonzero");
    const Real NX = intDchiX / (normF+EPS), NY = intDchiY / (normF+EPS);
    const Real PX = intPx/intNorm, PY = intPy/intNorm;
    const std::array<Real,2> org = info.pos<Real>(0, 0);
    const Real INDX = std::round(PX - org[0]) * invh;
    const Real INDY = std::round(PY - org[1]) * invh;
    assert(INDX>=0 && INDX<_BS_ && INDY>=0 && INDY<_BS_);
    const std::vector<Real> {NX, NY, PX, PY, INDX, INDY};
  };

  #pragma omp parallel for schedule(static)
  for (int i=0; i<N; ++i)
  {
    for (int j=0; j<N; ++j)
    {
      const auto& iBlocks = shapes[i]->obstacleBlocks;
      const auto& jBlocks = shapes[j]->obstacleBlocks;
      assert(iBlocks.size() == jBlocks.size());
      const size_t nBlocks = iBlocks.size();

      for (size_t k=0; k<nBlocks; ++k)
      {
        if ( iBlocks[k] == nullptr ) continue;
        if ( jBlocks[k] == nullptr ) continue;
        const auto collInfo = findIntersect(iBlocks[k], jBlocks[k], velInfo[k]);
        if ( collInfo.size() == 0 ) continue;
        assert(collInfo.size() == 6);

        const Real NX = collInfo[0], NY = collInfo[1]; // collision normal
        const Real CX = collInfo[2], CY = collInfo[3]; // collision point
        const int INDX = collInfo[4], INDY = collInfo[5]; // coll block index

        // compute linear, rotational, and total velocities for j and i
        const Real iUl = shapes[i]->u, iVl = shapes[i]->v, iW =shapes[i]->omega;
        const Real jUl = shapes[j]->u, jVl = shapes[j]->v, jW =shapes[j]->omega;
        const UDEFMAT & iUDEF = iBlocks[k]->udef, & jUDEF = jBlocks[k]->udef;
        const Real iDCx = CX - shapes[i]->centerOfMass[0];
        const Real iDCy = CY - shapes[i]->centerOfMass[1];
        const Real iUr = - iW * iDCy, iVr =   iW * iDCx;
        const Real jUr = - jW * (CY - shapes[j]->centerOfMass[1]);
        const Real jVr =   jW * (CX - shapes[j]->centerOfMass[0]);

        const Real iUtot = iUl + iUr + iUDEF[INDY][INDX][0];
        const Real iVtot = iVl + iVr + iUDEF[INDY][INDX][1];
        const Real jUtot = jUl + jUr + jUDEF[INDY][INDX][0];
        const Real jVtot = jVl + jVr + jUDEF[INDY][INDX][1];
        const Real invNormal = 1 / (NX * NX + NY * NY + EPS);
        // relVel of collider point (i) wrt to collided (j) point dot normal:
        const Real projVel = ((iUtot-jUtot)*NX + (iVtot-jVtot)*NY) * invNormal;
        if (projVel>=0) continue; // vel goes away from coll: no need to bounce
        // we correct both rotation and translation: compute contributions
        const Real projLin = (iUl-jUtot) * NX + (iVl-jVtot) * NY;
        const Real projRot = (iUr-jUtot) * NX + (iVr-jVtot) * NY;
        const Real facLin = - std::min(projLin, (Real) 0);
        const Real facRot = - std::min(projRot, (Real) 0);
        const Real Wlin = facLin / (facLin + facRot + EPS);
        const Real Wrot = facRot / (facLin + facRot + EPS);
        assert( facLin >= 0 && facRot >= 0 && Wlin >= && Wrot >= 0 );
        if ( projLin < 0 ) { // lin velocity goes towards collision: bounce back
          shapes[i]->u -= 2 * projVel * Wlin * NX;
          shapes[i]->v -= 2 * projVel * Wlin * NY;
        }
        if ( projRot < 0 ) { // rot velocity goes towards collision: bounce back
          const Real rdotr = iDCx * iDCx + iDCy * iDCy + EPS;
          const Real rcrossn = iDCx * NY - iDCy * NX;
          shapes[i]->omega -= 2 * projVel * Wrot * rcrossn / rdotr;
        }
      }
    }
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

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "UpdateObjects.h"
#include "../Shape.h"

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

void UpdateObjects::integrateMomenta(Shape * const shape) const
{
  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
  const Real lambdt = sim.lambda * sim.dt;
  const double hsq = std::pow(velInfo[0].h_gridpoint, 2);
  double _M = 0, _J = 0, UM = 0, VM = 0, AM = 0; //linear momenta

  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : _M,_J,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;

    if(OBLOCK[velInfo[i].blockID] == nullptr) continue;
    CHI_MAT & __restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
    UDEFMAT & __restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      const Real penalFac = hsq * chi[iy][ix]*lambdt / (1 + chi[iy][ix]*lambdt);
      double p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      const Real udiff[2] = {
        VEL(ix,iy).u[0] - udef[iy][ix][0],
        VEL(ix,iy).u[1] - udef[iy][ix][1]
      };
      _M += penalFac;
      UM += penalFac * udiff[0];
      VM += penalFac * udiff[1];
      _J += penalFac * (p[0]*p[0] + p[1]*p[1]);
      AM += penalFac * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }

  if(not shape->bForcedx)  shape->u     = UM / _M;
  if(not shape->bForcedy)  shape->v     = VM / _M;
  if(not shape->bBlockang) shape->omega = AM / _J;
}

void UpdateObjects::penalize(const double dt) const
{
  const Real lamdt = sim.lambda * dt;
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
      const Real alpha = 1/(1 + lamdt * X[iy][ix]);
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
  for(Shape * const shape : sim.shapes) integrateMomenta(shape);
  penalize(dt);
  sim.stopProfiler();


  // 2) update objects' velocities
  sim.startProfiler("Obj_update");
  //for(Shape * const shape : sim.shapes) shape->updateVelocity(dt);

  // 3) update simulation frame's velocity
  int nSum[2] = {0, 0};
  double uSum[2] = {0, 0};
  for(Shape * const shape : sim.shapes) shape->updateLabVelocity(nSum, uSum);
  if(nSum[0]>0) sim.uinfx = uSum[0]/nSum[0];
  if(nSum[1]>0) sim.uinfy = uSum[1]/nSum[1];

  // 4) update objects' centres of mass
  for(Shape * const shape : sim.shapes)
  {
    shape->updatePosition(dt);
    double p[2] = {0,0};
    shape->getCentroid(p);
    const Real maxExtent = std::max(sim.bpdx, sim.bpdy);
    double simExtent[2] = {sim.bpdx/maxExtent, sim.bpdy/maxExtent};
    if (p[0]<0 || p[0]>simExtent[0] || p[1]<0 || p[1]>simExtent[1]) {
      std::cout << "Body out of domain: " << p[0] << " " << p[1] << std::endl;
      exit(0);
    }
  }
  sim.stopProfiler();
}

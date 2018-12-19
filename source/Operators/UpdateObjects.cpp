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
//using UDEFMAT = Real[sizeY][sizeX][2];

void UpdateObjects::integrateForce(Shape * const shape) const
{
  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();
  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

  const double hsq = std::pow(forceInfo[0].h_gridpoint, 2);
  double _M = 0, _J = 0, UM = 0, VM = 0, AM = 0; //linear momenta

  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : _M,_J, UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ FPNL = *(VectorBlock*)forceInfo[i].ptrBlock;

    if(OBLOCK[forceInfo[i].blockID] == nullptr) continue;
    CHI_MAT & __restrict__ chi = OBLOCK[forceInfo[i].blockID]->chi;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      const Real chihs = chi[iy][ix] * hsq;
      if (chihs <= 0) continue;
      double p[2]; forceInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      _M += chihs;
      UM -= hsq * FPNL(ix,iy).u[0]; // force on object is minus
      VM -= hsq * FPNL(ix,iy).u[1]; // the force on the fluid!
      _J += chihs * (p[0]*p[0] + p[1]*p[1]);
      AM -= hsq * (p[0]*FPNL(ix,iy).u[1] - p[1]*FPNL(ix,iy).u[0]);
    }
  }

  shape->Fx = UM / _M;
  shape->Fy = VM / _M;
  shape->Tz = AM / _J;
}

void UpdateObjects::operator()(const double dt)
{
  // penalization force is now assumed to be finalized
  // 1) integrate momentum
  sim.startProfiler("Obj_force");
  for(Shape * const shape : sim.shapes) integrateForce(shape);
  sim.stopProfiler();

  // 2) update objects' velocities
  sim.startProfiler("Obj_update");
  for(Shape * const shape : sim.shapes) shape->updateVelocity(dt);

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

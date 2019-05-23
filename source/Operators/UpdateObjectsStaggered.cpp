//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "UpdateObjectsStaggered.h"
#include "../Shape.h"

using namespace cubism;

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

#define EXPL_INTEGRATE_MOM

void UpdateObjectsStaggered::integrateMomenta(Shape * const shape) const
{
  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
  const double hsq = std::pow(velInfo[0].h_gridpoint, 2);
  double PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  {
    static constexpr int stenBeg[3] = {0,0,0}, stenEnd[3] = {2,2,1};
    VectorLab velLab; velLab.prepare(*(sim.vel), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic,1)
    for(size_t i=0; i<Nblocks; i++)
    {
      if(OBLOCK[velInfo[i].blockID] == nullptr) continue;

      velLab.load(velInfo[i],0); const auto& __restrict__ VEL = velLab;
      const CHI_MAT & __restrict__ rho = OBLOCK[velInfo[i].blockID]->rho;
      const CHI_MAT & __restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
      #ifndef EXPL_INTEGRATE_MOM
        const Real lambdt = sim.lambda * sim.dt;
        const auto& __restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
      #endif

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if (chi[iy][ix] <= 0) continue;
        const Real U = (VEL(ix,iy).u[0] + VEL(ix+1,iy).u[0]) / 2;
        const Real V = (VEL(ix,iy).u[1] + VEL(ix,iy+1).u[1]) / 2;
        #ifdef EXPL_INTEGRATE_MOM
          const Real F = hsq * rho[iy][ix] * chi[iy][ix], udiff[2] = { U, V };
        #else
          const Real Xlamdt = chi[iy][ix] * lambdt;
          const Real F = hsq * rho[iy][ix] * Xlamdt / (1 + Xlamdt);
          const Real udiff[2] = { U - udef[iy][ix][0], V - udef[iy][ix][1] };
        #endif
        double p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
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

void UpdateObjectsStaggered::penalize(const double dt) const
{
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  const Real lamdt = sim.lambda * dt;//, dh = velInfo[0].h_gridpoint/2;
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1,0}, stenEnd[3] = {1,1,1};
    VectorLab udefLab; udefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);
    ScalarLab chiLab;   chiLab.prepare(*(sim.chi ), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic, 1)
    for (size_t i=0; i < Nblocks; i++)
    for (Shape * const shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
      if (o == nullptr) continue;

      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real US = shape->u, VS = shape->v, WS = shape->omega;

      const CHI_MAT & __restrict__ X = o->chi;
      //const UDEFMAT & __restrict__ udef = o->udef;
            auto& __restrict__    V = *(VectorBlock*) velInfo[i].ptrBlock;
      udefLab.load(uDefInfo[i],0); const auto& __restrict__ UDEF = udefLab;
      chiLab.load(chiInfo[i],0);  const auto& __restrict__ CHI = chiLab;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real CHIX = (CHI(ix,iy).s + CHI(ix-1,iy).s) / 2;
        const Real CHIY = (CHI(ix,iy).s + CHI(ix,iy-1).s) / 2;
        const Real CHIXO = (X[iy][ix]+(ix>0? X[iy][ix-1] : CHI(ix-1,iy).s) )/2;
        const Real CHIYO = (X[iy][ix]+(iy>0? X[iy-1][ix] : CHI(ix,iy-1).s) )/2;
        // What if multiple obstacles share a block? Do not penalize with this
        // obstacle if CHI stored on the grid is greater than obstacle's CHI.
        const Real penX = CHIX>CHIXO || CHIX<=0 ? 1 : 1/(1 +lamdt*CHIX);
        const Real penY = CHIY>CHIYO || CHIY<=0 ? 1 : 1/(1 +lamdt*CHIY);

        Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
        const Real UDEFX = (UDEF(ix,iy).u[0] + UDEF(ix-1,iy).u[0]) / 2;
        const Real UDEFY = (UDEF(ix,iy).u[1] + UDEF(ix,iy-1).u[1]) / 2;
        V(ix,iy).u[0] = penX*V(ix,iy).u[0] +(1-penX)*(US - WS*p[1] + UDEFX);
        V(ix,iy).u[1] = penY*V(ix,iy).u[1] +(1-penY)*(VS + WS*p[0] + UDEFY);
      }
    }
  }
}

void UpdateObjectsStaggered::operator()(const double dt)
{
  // penalization force is now assumed to be finalized
  // 1) integrate momentum
  sim.startProfiler("Obj_force");
  for(Shape * const shape : sim.shapes) {
    integrateMomenta(shape);
    shape->updateVelocity(dt);
  }
  penalize(dt);
  sim.stopProfiler();
}

//
//  OperatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "RKstep2.h"

static constexpr int sizeY = VectorBlock::sizeY;
static constexpr int sizeX = VectorBlock::sizeX;

static inline Real dU_adv_dif(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real upx = V(ix+1, iy).u[0], upy = V(ix, iy+1).u[0];
  const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
  const Real ulx = V(ix-1, iy).u[0], uly = V(ix, iy-1).u[0];
  const Real dUadv = (ucc+uinf[0]) * (upx-ulx) + (vcc+uinf[1]) * (upy-uly);
  const Real dUdif = upx + upy + ulx + uly - 4 *ucc;
  return advF * dUadv + difF * dUdif;
}

static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real vpx = V(ix+1, iy).u[1], vpy = V(ix, iy+1).u[1];
  const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
  const Real vlx = V(ix-1, iy).u[1], vly = V(ix, iy-1).u[1];
  const Real dVadv = (ucc+uinf[0]) * (vpx-vlx) + (vcc+uinf[1]) * (vpy-vly);
  const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
  return advF * dVadv + difF * dVdif;
}

void RKstep2::operator()(const double dt)
{
  sim.startProfiler("RKstep2");
  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};
  const Real dfac = sim.nu/h/h, afac = -0.5/h, divFac = 0.5*h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
    VectorLab tmplab; tmplab.prepare(*(sim.tmpV), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      tmplab.load(tmpVInfo[i], 0); // loads   vel field with ghosts
      const VectorLab & __restrict__ TMP = tmplab;
      VectorBlock & __restrict__ vel  = *(VectorBlock*) velInfo[i].ptrBlock;
      ScalarBlock & __restrict__ pRHS = *(ScalarBlock*)pRHSInfo[i].ptrBlock;

      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        pRHS(ix,0).s -= divFac * dV_adv_dif(TMP,UINF,afac,dfac,ix,-1);

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        pRHS(0,iy).s -= divFac * dU_adv_dif(TMP,UINF,afac,dfac,-1,iy);

        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          const Real dU = dU_adv_dif(TMP, UINF, afac, dfac, ix, iy);
          const Real dV = dV_adv_dif(TMP, UINF, afac, dfac, ix, iy);

          vel(ix,iy).u[0] = vel(ix,iy).u[0] + dt * dU;
          vel(ix,iy).u[1] = vel(ix,iy).u[1] + dt * dV;
          if(ix>      0) pRHS(ix-1,iy  ).s += divFac * dU;
          if(iy>      0) pRHS(ix  ,iy-1).s += divFac * dV;
          if(ix<sizeX-1) pRHS(ix+1,iy  ).s -= divFac * dU;
          if(iy<sizeY-1) pRHS(ix  ,iy+1).s -= divFac * dV;
        }

        pRHS(sizeX-1,iy).s += divFac * dU_adv_dif(TMP,UINF,afac,dfac,sizeX,iy);
      }

      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        pRHS(ix,sizeY-1).s += divFac * dV_adv_dif(TMP,UINF,afac,dfac,ix,sizeY);
    }
  }
  sim.stopProfiler();
}

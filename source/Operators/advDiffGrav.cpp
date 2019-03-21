//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiffGrav.h"
//#define DIV_ADVECT

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

void advDiffGrav::operator()(const double dt)
{
  sim.startProfiler("advDiffGrav");
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    const VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
          VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    T.copy(V);
  }
  //sim.dumpVel("step_"+std::to_string(sim.step)+"_");
  //sim.dumpTmpV("step_"+std::to_string(sim.step)+"_");

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  const std::array<Real,2>& G = sim.gravity;
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};
  const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab tmplab; tmplab.prepare(*(sim.tmpV), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      tmplab.load(tmpVInfo[i], 0); // loads   vel field with ghosts
      const VectorLab  & __restrict__   TMP = tmplab;
            VectorBlock& __restrict__     V =*(VectorBlock*)velInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ invRho=*(ScalarBlock*)rhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real gravFac = dt * (1 - invRho(ix,iy).s);
        const Real dUAdvDiff = dU_adv_dif(TMP,UINF,afac,dfac,ix,iy);
        const Real dVAdvDiff = dV_adv_dif(TMP,UINF,afac,dfac,ix,iy);
        V(ix,iy).u[0] = V(ix,iy).u[0] + dUAdvDiff + G[0]*gravFac;
        V(ix,iy).u[1] = V(ix,iy).u[1] + dVAdvDiff + G[1]*gravFac;
      }
    }
  }
  sim.stopProfiler();
}

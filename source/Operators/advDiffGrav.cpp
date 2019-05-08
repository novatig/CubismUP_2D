//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiffGrav.h"

using namespace cubism;

void advDiffGrav::operator()(const double dt)
{
  sim.startProfiler("advDiffGrav");
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
  const auto isW = [&](const BlockInfo& info) {
    return info.index[0] == 0;
  };
  const auto isE = [&](const BlockInfo& info) {
    return info.index[0] == sim.bpdx-1;
  };
  const auto isS = [&](const BlockInfo& info) {
    return info.index[1] == 0;
  };
  const auto isN = [&](const BlockInfo& info) {
    return info.index[1] == sim.bpdy-1;
  };

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  const std::array<Real,2>& G = sim.gravity;
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};
  const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ invRho=*(ScalarBlock*)rhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY && isW(velInfo[i]); ++iy) { // west
        V(BX-1, iy).u[0] = V(BX,iy).u[0]+UINF[0]>0? 0 : V(BX-1, iy).u[0];
        V(BX-1, iy).u[1] = V(BX,iy).u[0]+UINF[0]>0? 0 : V(BX-1, iy).u[1];
      }

      for(int ix=0; ix<VectorBlock::sizeX && isS(velInfo[i]); ++ix) { // south
        V(ix, BY-1).u[0] = V(ix,BY).u[1]+UINF[1]>0? 0 : V(ix, BY-1).u[0];
        V(ix, BY-1).u[1] = V(ix,BY).u[1]+UINF[1]>0? 0 : V(ix, BY-1).u[1];
      }

      for(int iy=0; iy<VectorBlock::sizeY && isE(velInfo[i]); ++iy) { // east
        V(EX+1, iy).u[0] = V(EX,iy).u[0]+UINF[0]<0? 0 : V(EX+1, iy).u[0];
        V(EX+1, iy).u[1] = V(EX,iy).u[0]+UINF[0]<0? 0 : V(EX+1, iy).u[1];
      }

      for(int ix=0; ix<VectorBlock::sizeX && isN(velInfo[i]); ++ix) { // north
        V(ix, EY+1).u[0] = V(ix,EY).u[1]+UINF[1]<0? 0 : V(ix, EY+1).u[0];
        V(ix, EY+1).u[1] = V(ix,EY).u[1]+UINF[1]<0? 0 : V(ix, EY+1).u[1];
      }

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real gravFac = dt * (1 - invRho(ix,iy).s);
        const Real upx = V(ix+1, iy).u[0], upy = V(ix, iy+1).u[0];
        const Real ulx = V(ix-1, iy).u[0], uly = V(ix, iy-1).u[0];
        const Real vpx = V(ix+1, iy).u[1], vpy = V(ix, iy+1).u[1];
        const Real vlx = V(ix-1, iy).u[1], vly = V(ix, iy-1).u[1];
        const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
        const Real dUdif = upx + upy + ulx + uly - 4 * ucc;
        const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
        #if 1
          const Real UE = (upx+ucc)/2, UW = (ulx+ucc)/2;
          const Real UN = (upy+ucc)/2, US = (uly+ucc)/2;
          const Real VE = (vpx+vcc)/2, VW = (vlx+vcc)/2;
          const Real VN = (vpy+vcc)/2, VS = (vly+vcc)/2;
          const Real dUadvX = (UE + UINF[0]) * UE - (UW + UINF[0]) * UW;
          const Real dUadvY = (VN + UINF[1]) * UN - (VS + UINF[1]) * US;
          const Real dVadvX = (UE + UINF[0]) * VE - (UW + UINF[0]) * VW;
          const Real dVadvY = (VN + UINF[1]) * VN - (VS + UINF[1]) * VS;
          const Real dUAdvDiff = 2*afac * (dUadvX + dUadvY) + dfac * dUdif;
          const Real dVAdvDiff = 2*afac * (dVadvX + dVadvY) + dfac * dVdif;
        #else
          const Real dUadv = (ucc+UINF[0])*(upx-ulx) + (vcc+UINF[1])*(upy-uly);
          const Real dVadv = (ucc+UINF[0])*(vpx-vlx) + (vcc+UINF[1])*(vpy-vly);
          const Real dUAdvDiff = afac * dUadv + dfac * dUdif;
          const Real dVAdvDiff = afac * dVadv + dfac * dVdif;
        #endif

        TMP(ix,iy).u[0] = V(ix,iy).u[0] + dUAdvDiff + G[0]*gravFac;
        TMP(ix,iy).u[1] = V(ix,iy).u[1] + dVAdvDiff + G[1]*gravFac;
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
          VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
  }
  sim.stopProfiler();
}

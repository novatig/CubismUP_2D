//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiffGrav.h"

using namespace cubism;

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
      vellab.load(velInfo[i], 0); const VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ invRho=*(ScalarBlock*)rhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real gravFac = dt * (1 - invRho(ix,iy).s);
        const Real dUAdvDiff = dU_adv_dif(V, UINF, afac, dfac, ix, iy);
        const Real dVAdvDiff = dV_adv_dif(V, UINF, afac, dfac, ix, iy);
        TMP(ix,iy).u[0] = V(ix,iy).u[0] + dUAdvDiff + G[0]*gravFac;
        TMP(ix,iy).u[1] = V(ix,iy).u[1] + dVAdvDiff + G[1]*gravFac;
      }

      if ( isW(velInfo[i]) ) // west
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        {
          const Real uAdv = V(BX,iy).u[0]+UINF[0], vAdv = V(BX,iy).u[1]+UINF[1];
          const Real upx = V(BX+1, iy).u[0], upy = V(BX, iy+1).u[0];
          const Real ulx = uAdv>0? 0 : V(BX-1, iy).u[0], uly = V(BX, iy-1).u[0];
          const Real vpx = V(BX+1, iy).u[1], vpy = V(BX, iy+1).u[1];
          const Real vlx = uAdv>0? 0 : V(BX-1, iy).u[1], vly = V(BX, iy-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(BX,iy).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(BX,iy).u[1]);
          TMP(BX,iy).u[0]= V(BX,iy).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(BX,iy).u[1]= V(BX,iy).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      if( isE(velInfo[i]) ) // east
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        {
          const Real uAdv = V(EX,iy).u[0]+UINF[0], vAdv = V(EX,iy).u[1]+UINF[1];
          const Real upx = uAdv<0? 0 : V(EX+1, iy).u[0], upy = V(EX, iy+1).u[0];
          const Real ulx = V(EX-1, iy).u[0], uly = V(EX, iy-1).u[0];
          const Real vpx = uAdv<0? 0 : V(EX+1, iy).u[1], vpy = V(EX, iy+1).u[1];
          const Real vlx = V(EX-1, iy).u[1], vly = V(EX, iy-1).u[1];
          // if momentum is advected inwards then dirichlet, otherwise outflow
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(EX,iy).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(EX,iy).u[1]);
          TMP(EX,iy).u[0]= V(EX,iy).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(EX,iy).u[1]= V(EX,iy).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      if( isS(velInfo[i]) ) // south
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          const Real uAdv = V(ix,BY).u[0]+UINF[0], vAdv = V(ix,BY).u[1]+UINF[1];
          const Real upx = V(ix+1, BY).u[0], upy = V(ix, BY+1).u[0];
          const Real ulx = V(ix-1, BY).u[0], uly = vAdv>0? 0 : V(ix, BY-1).u[0];
          const Real vpx = V(ix+1, BY).u[1], vpy = V(ix, BY+1).u[1];
          const Real vlx = V(ix-1, BY).u[1], vly = vAdv>0? 0 : V(ix, BY-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(ix,BY).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(ix,BY).u[1]);
          TMP(ix,BY).u[0]= V(ix,BY).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(ix,BY).u[1]= V(ix,BY).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      if( isN(velInfo[i]) ) // north
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          const Real uAdv = V(ix,EY).u[0]+UINF[0], vAdv = V(ix,EY).u[1]+UINF[1];
          const Real upx = V(ix+1, EY).u[0], upy = vAdv<0? 0 : V(ix, EY+1).u[0];
          const Real ulx = V(ix-1, EY).u[0], uly = V(ix, EY-1).u[0];
          const Real vpx = V(ix+1, EY).u[1], vpy = vAdv<0? 0 : V(ix, EY+1).u[1];
          const Real vlx = V(ix-1, EY).u[1], vly = V(ix, EY-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(ix,EY).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(ix,EY).u[1]);
          TMP(ix,EY).u[0]= V(ix,EY).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(ix,EY).u[1]= V(ix,EY).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      // fix last corners
      if ( isW(velInfo[i]) && isS(velInfo[i]) ) // west
        {
          const Real uAdv = V(BX,BY).u[0]+UINF[0], vAdv = V(BX,BY).u[1]+UINF[1];
          const Real upx = V(BX+1, BY).u[0], upy = V(BX, BY+1).u[0];
          const Real ulx = uAdv>0? 0 : V(BX-1, BY).u[0];
          const Real uly = vAdv>0? 0 : V(BX, BY-1).u[0];
          const Real vpx = V(BX+1, BY).u[1], vpy = V(BX, BY+1).u[1];
          const Real vlx = uAdv>0? 0 : V(BX-1, BY).u[1];
          const Real vly = vAdv>0? 0 : V(BX, BY-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(BX,BY).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(BX,BY).u[1]);
          TMP(BX,BY).u[0]= V(BX,BY).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(BX,BY).u[1]= V(BX,BY).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      if ( isW(velInfo[i]) && isN(velInfo[i]) )
        {
          const Real uAdv = V(BX,EY).u[0]+UINF[0], vAdv = V(BX,EY).u[1]+UINF[1];
          const Real upx = V(BX+1, EY).u[0], upy = vAdv<0? 0 : V(BX, EY+1).u[0];
          const Real ulx = uAdv>0? 0 : V(BX-1, EY).u[0], uly = V(BX, EY-1).u[0];
          const Real vpx = V(BX+1, EY).u[1], vpy = vAdv<0? 0 : V(BX, EY+1).u[1];
          const Real vlx = uAdv>0? 0 : V(BX-1, EY).u[1], vly = V(BX, EY-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(BX,EY).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(BX,EY).u[1]);
          TMP(BX,EY).u[0]= V(BX,EY).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(BX,EY).u[1]= V(BX,EY).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      if ( isE(velInfo[i]) && isS(velInfo[i]) )
        {
          const Real uAdv = V(EX,BY).u[0]+UINF[0], vAdv = V(EX,BY).u[1]+UINF[1];
          const Real upx = uAdv<0? 0 : V(EX+1, BY).u[0], upy = V(EX, BY+1).u[0];
          const Real ulx = V(EX-1, BY).u[0], uly = vAdv>0? 0 : V(EX, BY-1).u[0];
          const Real vpx = uAdv<0? 0 : V(EX+1, BY).u[1], vpy = V(EX, BY+1).u[1];
          const Real vlx = V(EX-1, BY).u[1], vly = vAdv>0? 0 : V(EX, BY-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(EX,BY).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(EX,BY).u[1]);
          TMP(EX,BY).u[0]= V(EX,BY).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(EX,BY).u[1]= V(EX,BY).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
        }

      if ( isE(velInfo[i]) && isN(velInfo[i]) )
        {
          const Real uAdv = V(EX,EY).u[0]+UINF[0], vAdv = V(EX,EY).u[1]+UINF[1];
          const Real upx = uAdv<0? 0 : V(EX+1, EY).u[0];
          const Real upy = vAdv<0? 0 : V(EX, EY+1).u[0];
          const Real ulx = V(EX-1, EY).u[0], uly = V(EX, EY-1).u[0];
          const Real vpx = uAdv<0? 0 : V(EX+1, EY).u[1];
          const Real vpy = vAdv<0? 0 : V(EX, EY+1).u[1];
          const Real vlx = V(EX-1, EY).u[1], vly = V(EX, EY-1).u[1];
          const Real gradUx = upx-ulx, gradUy = upy-uly;
          const Real gradVx = vpx-vlx, gradVy = vpy-vly;
          const Real dUdif = dfac * (upx+upy+ulx+uly - 4*V(EX,EY).u[0]);
          const Real dVdif = dfac * (vpx+vpy+vlx+vly - 4*V(EX,EY).u[1]);
          TMP(EX,EY).u[0]= V(EX,EY).u[0] +afac*(uAdv*gradUx+vAdv*gradUy) +dUdif;
          TMP(EX,EY).u[1]= V(EX,EY).u[1] +afac*(uAdv*gradVx+vAdv*gradVy) +dVdif;
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

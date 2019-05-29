//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiffGravStaggered.h"
//#define NOINFLOW

using namespace cubism;
static constexpr double EPS = std::numeric_limits<Real>::epsilon();
static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 1, 1, 1};
static constexpr int stenBegV[3] = {-1,-1, 0}, stenEndV[3] = { 2, 2, 1};

void advDiffGravStaggered::operator()(const double dt)
{
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0; };
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
  const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0; };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  const Real G[2]= { dt*sim.gravity[0], dt*sim.gravity[1] };
  const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeXW = 1 - std::pow( std::max(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeYS = 1 - std::pow( std::max(UINF[1],(Real) 0) / norUinf, 2);
  const Real fadeXE = 1 - std::pow( std::min(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeYN = 1 - std::pow( std::min(UINF[1],(Real) 0) / norUinf, 2);
  sim.startProfiler("advDiffGrav");

  #pragma omp parallel
  {
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBegV, stenEndV, 1);
    ScalarLab rholab; rholab.prepare(*(sim.invRho), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; ++i)
    {
      vellab.load( velInfo[i], 0); auto & __restrict__ V = vellab;
      rholab.load(iRhoInfo[i], 0); auto & __restrict__ IRHO = rholab;
      auto& __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      for(int iy=-1; iy<=VectorBlock::sizeY && isE(velInfo[i]); ++iy) { // west
        V(EX+1,iy).u[0] = V(EX  ,iy).u[0];
        V(EX+1,iy).u[1] = V(EX-1,iy).u[1];
        V(EX  ,iy).u[1] = V(EX-1,iy).u[1];
      }

      for(int ix=-1; ix<=VectorBlock::sizeX && isN(velInfo[i]); ++ix) { // north
        V(ix,EY+1).u[1] = V(ix,EY  ).u[1];
        V(ix,EY+1).u[0] = V(ix,EY-1).u[0];
        V(ix,EY  ).u[0] = V(ix,EY-1).u[0];
      }

      if( isE(velInfo[i]) && isN(velInfo[i]) ) {
        V(EX  ,EY  ).u[0] = V(EX,EY-1).u[0];
        V(EX  ,EY+1).u[0] = V(EX,EY-1).u[0];
        V(EX+1,EY  ).u[0] = V(EX,EY-1).u[0];
        V(EX+1,EY+1).u[0] = V(EX,EY-1).u[0];

        V(EX  ,EY  ).u[1] = V(EX-1,EY).u[1];
        V(EX+1,EY  ).u[1] = V(EX-1,EY).u[1];
        V(EX  ,EY+1).u[1] = V(EX-1,EY).u[1];
        V(EX+1,EY+1).u[1] = V(EX-1,EY).u[1];
      }
      if( isW(velInfo[i]) && isN(velInfo[i]) ) {
        V(BX  ,EY  ).u[0] = V(BX,EY-1).u[0];
        V(BX  ,EY+1).u[0] = V(BX,EY-1).u[0];
        V(BX-1,EY  ).u[0] = V(BX,EY-1).u[0];
        V(BX-1,EY+1).u[0] = V(BX,EY-1).u[0];

        V(BX-1,EY+1).u[1] = V(BX,EY).u[1];
      }
      if( isE(velInfo[i]) && isS(velInfo[i]) ) {
        V(EX+1,BY-1).u[0] = V(EX,BY).u[0];

        V(EX  ,BY  ).u[1] = V(EX-1,BY).u[1];
        V(EX+1,BY  ).u[1] = V(EX-1,BY).u[1];
        V(EX  ,BY-1).u[1] = V(EX-1,BY).u[1];
        V(EX+1,BY-1).u[1] = V(EX-1,BY).u[1];
      }
      if( isW(velInfo[i]) && isS(velInfo[i]) ) {
        V(BX-1,BY-1).u[0] = V(BX,BY).u[0];
        V(BX-1,BY-1).u[1] = V(BX,BY).u[1];
      }

      for(int iy=-1; iy<=VectorBlock::sizeY && isW(velInfo[i]); ++iy) { // west
        V(BX-1,iy).u[0] *= fadeXW; V(BX-1,iy).u[1] *= fadeXW;
        //V(BX  ,iy).u[0] *= fadeXW; //V(BX  ,iy).u[1] *= fadeXW;
      }

      for(int ix=-1; ix<=VectorBlock::sizeX && isS(velInfo[i]); ++ix) { //south
        V(ix, BY-1).u[0] *= fadeYS; V(ix, BY-1).u[1] *= fadeYS;
        //V(ix, BY  ).u[0] *= fadeYS;
        //V(ix, BY  ).u[1] *= fadeYS;
      }

      for(int iy=-1; iy<=VectorBlock::sizeY && isE(velInfo[i]); ++iy) { // west
        V(EX+1,iy).u[0] *= fadeXE; V(EX+1,iy).u[1] *= fadeXE;
        // //V(EX  ,iy).u[0] *= fadeXE;
        // V(EX  ,iy).u[1] *= fadeXE;
        // //V(EX-1,iy).u[1] *= fadeXE;
      }

      for(int ix=-1; ix<=VectorBlock::sizeX && isN(velInfo[i]); ++ix) { //south
        V(ix, EY+1).u[0] *= fadeYN; V(ix, EY+1).u[1] *= fadeYN;
        // V(ix, EY  ).u[0] *= fadeYN;
        // //V(ix, EY  ).u[1] *= fadeYN;
        // //V(ix, EY-1).u[0] *= fadeXN;
      }

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real gravFacU = G[0] * ( 1 - (IRHO(ix-1,iy).s+IRHO(ix,iy).s)/2 );
        const Real gravFacV = G[1] * ( 1 - (IRHO(ix,iy-1).s+IRHO(ix,iy).s)/2 );
        const Real upx = V(ix+1, iy).u[0], upy = V(ix, iy+1).u[0];
        const Real ulx = V(ix-1, iy).u[0], uly = V(ix, iy-1).u[0];
        const Real vpx = V(ix+1, iy).u[1], vpy = V(ix, iy+1).u[1];
        const Real vlx = V(ix-1, iy).u[1], vly = V(ix, iy-1).u[1];
        const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
        #if 1
        {
          const Real VadvU = (vpy + V(ix-1,iy+1).u[1] + vcc + vlx)/4  + UINF[1];
          const Real dUadv = (ucc+UINF[0]) * (upx-ulx) + VadvU * (upy - uly);
          const Real dUdif = upx + upy + ulx + uly - 4 * ucc;
          TMP(ix,iy).u[0] = V(ix,iy).u[0] + afac*dUadv + dfac*dUdif + gravFacU;
        }
        {
          const Real UadvV = (upx + V(ix+1,iy-1).u[0] + ucc + uly)/4 + UINF[0];
          const Real dVadv = UadvV * (vpx-vlx) + (vcc+UINF[1]) * (vpy-vly);
          const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
          TMP(ix,iy).u[1] = V(ix,iy).u[1] + afac*dVadv + dfac*dVdif + gravFacV;
        }
        #else
        {
          const Real UE = (upx+ucc)/2, UW = (ulx+ucc)/2;
          const Real UN = (upy+ucc)/2, US = (uly+ucc)/2;
          const Real VN = (vpy+V(ix-1,iy+1).u[1])/2, VS = (vcc+vlx)/2;
          const Real dUadvX =  (UE + UINF[0]) * UE - (UW + UINF[0]) * UW;
          const Real dUadvY =  (VN + UINF[1]) * UN - (VS + UINF[1]) * US;
          const Real dUdif = upx + upy + ulx + uly - 4 * ucc;
          const Real dUAdvDiff = afac*(dUadvX + dUadvY) + dfac*dUdif;
          TMP(ix,iy).u[0] = V(ix,iy).u[0] + dUAdvDiff + gravFacU;
        }
        {
          const Real VE = (vpx+vcc)/2, VW = (vlx+vcc)/2;
          const Real VN = (vpy+vcc)/2, VS = (vly+vcc)/2;
          const Real UE = (upx+V(ix+1,iy-1).u[0])/2, UW = (ucc+uly)/2;
          const Real dVadvX =  (UE + UINF[0]) * VE - (UW + UINF[0]) * VW;
          const Real dVadvY =  (VN + UINF[1]) * VN - (VS + UINF[1]) * VS;
          const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
          const Real dVAdvDiff = afac*(dVadvX + dVadvY) + dfac*dVdif;
          TMP(ix,iy).u[1] = V(ix,iy).u[1] + dVAdvDiff + gravFacV;
        }
        #endif
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; ++i) {
          VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
  }


  if (1)
  {
    ////////////////////////////////////////////////////////////////////////////
    Real IF = 0, AF = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+ : IF, AF)
    for (size_t i=0; i < Nblocks; i++) {
      VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
      for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy) {
        IF -= V(BX,iy).u[0]; AF += std::fabs(V(BX,iy).u[0]);
      }
      for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy) {
        IF += V(EX,iy).u[0]; AF += std::fabs(V(EX,iy).u[0]);
      }
      for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix) {
        IF -= V(ix,BY).u[1]; AF += std::fabs(V(ix,BY).u[1]);
      }
      for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix) {
        IF += V(ix,EY).u[1]; AF += std::fabs(V(ix,EY).u[1]);
      }
      if( isN(velInfo[i]) && isW(velInfo[i]) ) {
        IF += V(BX,EY).u[0]; AF -= std::fabs(V(BX,EY).u[0]);
      }
      if( isN(velInfo[i]) && isE(velInfo[i]) ) {
        IF -= V(EX,EY).u[0]; AF -= std::fabs(V(EX,EY).u[0]);
        IF -= V(EX,EY).u[1]; AF -= std::fabs(V(EX,EY).u[1]);
      }
      if( isS(velInfo[i]) && isE(velInfo[i]) ) {
        IF += V(EX,BY).u[1]; AF -= std::fabs(V(EX,BY).u[1]);
      }
    }
    ////////////////////////////////////////////////////////////////////////////
    const Real corr = IF/std::max(AF, EPS);
    //const Real corr = IF/( 2*(BSY*sim.bpdy -1) + 2*(BSX*sim.bpdx -1) );
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i < Nblocks; i++) {
      VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
      for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy)
        V(BX,iy).u[0] += corr * std::fabs(V(BX,iy).u[0]);
      for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy)
        V(EX,iy).u[0] -= corr * std::fabs(V(EX,iy).u[0]);
      for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix)
        V(ix,BY).u[1] += corr * std::fabs(V(ix,BY).u[1]);
      for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix)
        V(ix,EY).u[1] -= corr * std::fabs(V(ix,EY).u[1]);
    }
  }

  sim.stopProfiler();
}

/*

if(isW && isS) {
  const Real u = V(BX,BY).u[0]+UINF[0], v = V(BX,BY).u[1]+UINF[1];
  V(BX-1,BY).u[1] = u>0? 0 : V(BX-1,BY).u[1];
  V(BX,BY-1).u[0] = v>0? 0 : V(BX,BY-1).u[0];
  V(BX-1,BY-1).u[0] = 0; V(BX-1,BY-1).u[1] = 0;
}
if(isE && isS) {
  const Real u = V(EX,BY).u[0]+UINF[0], v = V(EX,BY).u[1]+UINF[1];
  V(EX+1,BY).u[1] = u<0? 0 : V(EX+1,BY).u[1];
  V(EX,BY-1).u[0] = v>0? 0 : V(EX,BY-1).u[0];
}
if(isW && isN) {
  const Real u = V(BX,EY).u[0]+UINF[0], v = V(BX,EY).u[1]+UINF[1];
  V(BX-1,EY).u[1] = u>0? 0 : V(BX-1,EY).u[1];
  //V(BX  ,EY).u[1] = u>0? 0 : V(BX  ,EY).u[1];
  V(BX,EY+1).u[0] = v<0? 0 : V(BX,EY+1).u[0];
  //V(BX,EY  ).u[0] = v<0? 0 : V(BX,EY  ).u[0];
}
if(isE && isN) {
  const Real u = V(EX,EY).u[0]+UINF[0], v = V(EX,EY).u[1]+UINF[1];
  V(EX+1,EY).u[1] = u<0? 0 : V(EX+1,EY).u[1];
  //V(EX  ,EY).u[1] = u<0? 0 : V(EX  ,EY).u[1];
  V(EX,EY+1).u[0] = v<0? 0 : V(EX,EY+1).u[0];
  //V(EX,EY  ).u[0] = v<0? 0 : V(EX,EY  ).u[0];
}

for(int iy=0; iy<=VectorBlock::sizeY && isW; ++iy) { // west
  const Real uAdvV = (V(BX,iy).u[0] + V(BX,iy-1).u[0])/2 +UINF[0];
  V(BX-1, iy).u[1] = uAdvV>0? 0 : V(BX-1, iy).u[1];
  //V(BX  , iy).u[1] = uAdvV>0? 0 : V(BX  , iy).u[1];
}
for(int ix=0; ix<=VectorBlock::sizeX && isS; ++ix) { //south
  const Real vAdvU = (V(ix,BY).u[1] + V(ix-1,BY).u[1])/2 +UINF[1];
  V(ix, BY-1).u[0] = vAdvU>0? 0 : V(ix, BY-1).u[0];
  //V(ix, BY  ).u[0] = vAdvU>0? 0 : V(ix, BY  ).u[0];
}
for(int iy=0; iy<=VectorBlock::sizeY && isE; ++iy) { //east
  const Real uAdvV = (V(EX+1, iy).u[0] + V(EX+1, iy-1).u[0])/2 +UINF[0];
  V(EX+1, iy).u[1] = uAdvV<0? 0 : V(EX+1, iy).u[1];
  //V(EX  , iy).u[1] = uAdvV<0? 0 : V(EX  , iy).u[1];
}
for(int ix=0; ix<=VectorBlock::sizeX && isN; ++ix) { //north
  const Real vAdvU = (V(ix, EY+1).u[1] + V(ix-1, EY+1).u[1])/2 +UINF[1];
  V(ix, EY+1).u[0] = vAdvU<0? 0 : V(ix, EY+1).u[0];
  //V(ix, EY  ).u[0] = vAdvU<0? 0 : V(ix, EY  ).u[0];
}

if(inflowDir == 0) // kill inflow from the y sides:
{
  if(inflowSide==0)
  {
    for(int iy=-1; iy<=VectorBlock::sizeY && isE; ++iy) { // east
      V(EX+1,iy).u[0] = V(EX,iy).u[0] - corrE;
      V(EX,iy).u[0]   = V(EX,iy).u[0] - corrE;
    }
    for(int iy=-1; iy<=VectorBlock::sizeY && isW; ++iy) { // west
      V(BX-1,iy).u[0] = 0; V(BX  ,iy).u[0] = 0;
      V(BX-1,iy).u[1] = 0; V(BX  ,iy).u[1] = 0;
    }
  }
  else
  {
    for(int iy=-1; iy<=VectorBlock::sizeY && isE; ++iy) { // east
      V(EX+1,iy).u[0] = 0; V(EX,iy).u[0]   = 0;
      V(EX+1,iy).u[1] = 0; V(EX,iy).u[1]   = 0;
    }
    for(int iy=-1; iy<=VectorBlock::sizeY && isW; ++iy) { // west
      V(BX-1,iy).u[0] = V(BX,iy).u[0] - corrW;
      V(BX  ,iy).u[0] = V(BX,iy).u[0] - corrW;
    }
  }

  for(int ix=-1; ix<=VectorBlock::sizeX && isS; ++ix) { // south
    V(ix, BY-1).u[1] = 0;
    V(ix, BY  ).u[1] = 0;
  }
  for(int ix=-1; ix<=VectorBlock::sizeX && isN; ++ix) { // north
    V(ix, EY+1).u[1] = 0;
  }
}
else
{
  for(int ix=-1; ix<=VectorBlock::sizeX && isS; ++ix) { // S
    V(ix,BY-1).u[1]= (V(ix,BY).u[1]+UINF[1]>0? 0: V(ix,BY-1).u[1]);
    V(ix,BY  ).u[1]= (V(ix,BY).u[1]+UINF[1]>0? 0: V(ix,BY  ).u[1]);
  }
  for(int ix=-1; ix<=VectorBlock::sizeX && isN; ++ix) { // N
    V(ix,EY+1).u[1]= (V(ix,EY).u[1]+UINF[1]<0? 0: V(ix,EY+1).u[1]);
  }
}

if(inflowDir == 1) // kill inflow from the x sides:
{
  for(int iy=-1; iy<=VectorBlock::sizeY && isW; ++iy) { // west
    V(BX-1, iy).u[0] = 0; V(BX  , iy).u[0] = 0;
  }
  for(int iy=-1; iy<=VectorBlock::sizeY && isE; ++iy) { // east
    V(EX, iy).u[0] = 0; V(EX+1, iy).u[0] = 0;
  }
}
else
{

}
*/

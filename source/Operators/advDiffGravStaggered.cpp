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
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
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
  const Real G[2]= { (Real) dt * sim.gravity[0], (Real) dt * sim.gravity[1] };
  const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeW= 1-std::pow(std::max(UINF[0],(Real)0)/norUinf,2)/BC_KILL_FAC;
  const Real fadeS= 1-std::pow(std::max(UINF[1],(Real)0)/norUinf,2)/BC_KILL_FAC;
  const Real fadeE= 1-std::pow(std::min(UINF[0],(Real)0)/norUinf,2)/BC_KILL_FAC;
  const Real fadeN= 1-std::pow(std::min(UINF[1],(Real)0)/norUinf,2)/BC_KILL_FAC;
  const auto fade = [&](VectorElement&B,const Real F) { B.u[0]*=F; B.u[1]*=F; };

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

      for(int iy=-1; iy<=BSY && isW(velInfo[i]); ++iy) fade(V(BX-1,iy), fadeW);
      for(int ix=-1; ix<=BSX && isS(velInfo[i]); ++ix) fade(V(ix,BY-1), fadeS);
      for(int iy=-1; iy<=BSY && isE(velInfo[i]); ++iy) fade(V(EX+1,iy), fadeE);
      for(int ix=-1; ix<=BSX && isN(velInfo[i]); ++ix) fade(V(ix,EY+1), fadeN);

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
    //const Real corr = IF/std::max(AF, EPS);
    const Real corr = IF/( 2*(BSY*sim.bpdy -1) + 2*(BSX*sim.bpdx -1) );
    printf("Relative inflow correction %e\n",corr);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i < Nblocks; i++) {
      VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
      for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy)
        V(BX,iy).u[0] += corr ;//* std::fabs(V(BX,iy).u[0]);
      for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy)
        V(EX,iy).u[0] -= corr ;//* std::fabs(V(EX,iy).u[0]);
      for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix)
        V(ix,BY).u[1] += corr ;//* std::fabs(V(ix,BY).u[1]);
      for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix)
        V(ix,EY).u[1] -= corr ;//* std::fabs(V(ix,EY).u[1]);
    }
  }

  sim.stopProfiler();
}

/*
  Real MX[2] = {0}, AX[2] = {0}, MY[2] = {0}, AY[2] = {0};
  #pragma omp parallel for schedule(dynamic) reduction(+ : MX[:2], AX[:2], \
                                                           MY[:2], AY[:2])
  for (size_t i=0; i < Nblocks; i++) {
    VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
    // handle corners outside of staggered grid, will be overwritten later
    if(isN(velInfo[i]) && isW(velInfo[i])) V(BX,EY).u[0] = 0;
    if(isN(velInfo[i]) && isE(velInfo[i])) { V(EX,EY).u[0]=0; V(EX,EY).u[1]=0; }
    if(isS(velInfo[i]) && isE(velInfo[i])) V(EX,BY).u[1] = 0;

    for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy) {
      MX[0] += V(BX,iy).u[0]; AX[0] += std::fabs( V(BX,iy).u[0] );
      MX[1] += V(BX,iy).u[1]; AX[1] += std::fabs( V(BX,iy).u[1] );
    }
    for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy) {
      MX[0] += V(EX,iy).u[0]; AX[0] += std::fabs( V(EX,iy).u[0] );
      MX[1] += V(EX,iy).u[1]; AX[1] += std::fabs( V(EX,iy).u[1] );
    }
    for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix) {
      MY[0] += V(ix,BY).u[0]; AY[0] += std::fabs( V(ix,BY).u[0] );
      MY[1] += V(ix,BY).u[1]; AY[1] += std::fabs( V(ix,BY).u[1] );
    }
    for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix) {
      MY[0] += V(ix,EY).u[0]; AY[0] += std::fabs( V(ix,EY).u[0] );
      MY[1] += V(ix,EY).u[1]; AY[1] += std::fabs( V(ix,EY).u[1] );
    }
  }
  AX[0] = std::max(EPS, AX[0]); AX[1] = std::max(EPS, AX[1]);
  AY[0] = std::max(EPS, AY[0]); AY[1] = std::max(EPS, AY[1]);
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeW = std::pow( std::max(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeS = std::pow( std::max(UINF[1],(Real) 0) / norUinf, 2);
  const Real fadeE = std::pow( std::min(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeN = std::pow( std::min(UINF[1],(Real) 0) / norUinf, 2);
  const Real fadeXW = fadeW * MX[0] / AX[0], fadeYW = fadeW * MX[1] / AX[1];
  const Real fadeXE = fadeE * MX[0] / AX[0], fadeYE = fadeE * MX[1] / AX[1];
  const Real fadeXS = fadeS * MY[0] / AY[0], fadeYS = fadeS * MY[1] / AY[1];
  const Real fadeXN = fadeN * MY[0] / AY[0], fadeYN = fadeN * MY[1] / AY[1];
  printf("fade %e %e - %e %e - %e %e - %e %e\n",fadeXW,fadeYW,fadeXE,fadeYE,fadeXS,fadeYS,fadeXN,fadeYN);
  const auto fade = [&] (VectorElement&B, const Real fadeX, const Real fadeY) {
    B.u[0] -= fadeX * std::fabs(B.u[0]); B.u[1] -= fadeY * std::fabs(B.u[1]);
  };

  for(int iy=-1; iy<=BSY && isW(velInfo[i]); ++iy)
    fade(V(BX-1,iy), fadeXW, fadeYW);

  for(int ix=-1; ix<=BSX && isS(velInfo[i]); ++ix)
    fade(V(ix,BY-1), fadeXS, fadeYS);

  for(int iy=-1; iy<=BSY && isE(velInfo[i]); ++iy)
    fade(V(EX+1,iy), fadeXE, fadeYE);

  for(int ix=-1; ix<=BSX && isN(velInfo[i]); ++ix)
    fade(V(ix,EY+1), fadeXN, fadeYN);
*/

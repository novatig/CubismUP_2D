//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiff.h"

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

void advDiff::operator()(const double dt)
{
  sim.startProfiler("advDiff");
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr double EPS = std::numeric_limits<Real>::epsilon();
  static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
  const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
  const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};
  const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeXW = 1 - std::pow( std::max(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeYS = 1 - std::pow( std::max(UINF[1],(Real) 0) / norUinf, 2);
  const Real fadeXE = 1 - std::pow( std::min(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeYN = 1 - std::pow( std::min(UINF[1],(Real) 0) / norUinf, 2);

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      for(int iy=-1; iy<=BSY && isW(velInfo[i]); ++iy) { // west
        V(BX-1,iy).u[0] *= fadeXW; V(BX-1,iy).u[1] *= fadeXW;
        //V(BX  ,iy).u[0] *= fadeXW; //V(BX  ,iy).u[1] *= fadeXW;
      }

      for(int ix=-1; ix<=BSX && isS(velInfo[i]); ++ix) { // south
        V(ix, BY-1).u[0] *= fadeYS; V(ix, BY-1).u[1] *= fadeYS;
        //V(ix, BY  ).u[0] *= fadeYS;
        //V(ix, BY  ).u[1] *= fadeYS;
      }

      for(int iy=-1; iy<=BSY && isE(velInfo[i]); ++iy) { // west
        V(EX+1,iy).u[0] *= fadeXE; V(EX+1,iy).u[1] *= fadeXE;
        // //V(EX  ,iy).u[0] *= fadeXE;
        // V(EX  ,iy).u[1] *= fadeXE;
        // //V(EX-1,iy).u[1] *= fadeXE;
      }

      for(int ix=-1; ix<=BSX && isN(velInfo[i]); ++ix) { // south
        V(ix, EY+1).u[0] *= fadeYN; V(ix, EY+1).u[1] *= fadeYN;
        // V(ix, EY  ).u[0] *= fadeYN;
        // //V(ix, EY  ).u[1] *= fadeYN;
        // //V(ix, EY-1).u[0] *= fadeXN;
      }

      for(int iy=0; iy<BSY; ++iy) for(int ix=0; ix<BSX; ++ix)
      {
        TMP(ix,iy).u[0] = V(ix,iy).u[0] + dU_adv_dif(V,UINF,afac,dfac,ix,iy);
        TMP(ix,iy).u[1] = V(ix,iy).u[1] + dV_adv_dif(V,UINF,afac,dfac,ix,iy);
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

  {
    ////////////////////////////////////////////////////////////////////////////
    Real IF = 0, AF = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+ : IF, AF)
    for (size_t i=0; i < Nblocks; i++) {
      const VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;
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
    }
    ////////////////////////////////////////////////////////////////////////////
    const Real corr = IF/std::max(AF, EPS);
    printf("Relative inflow correction %e\n",corr);
    //const Real corr = IF/( 2*(BSY*sim.bpdy -1) + 2*(BSX*sim.bpdx -1) );
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i < Nblocks; i++) {
      VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;
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

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiff_RK.h"

#define DIV_ADVECT
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

static inline Real divVel(const VectorLab&V, const Real fac, const int ix, const int iy)
{
  const Real upx = V(ix+1, iy).u[0], vpy = V(ix, iy+1).u[1];
  const Real ulx = V(ix-1, iy).u[0], vly = V(ix, iy-1).u[1];
  return fac * ((upx - ulx) + (vpy - vly));
}

void advDiff_RK::operator()(const double dt)
{
  sim.startProfiler("RKstep1");
  step1(dt);
  sim.stopProfiler();
  sim.startProfiler("RKstep2");
  step2(dt);
  sim.stopProfiler();
}

void advDiff_RK::step1(const double dt)
{
  static constexpr int stenBeg[3] = {-1, -1, 0}, stenEnd[3] = { 2,  2, 1};
  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};

  const Real dfac = (sim.nu/h)*(0.5*dt/h), afac = -0.5*dt/(2*h);
  const Real divFac = h/dt/2, pfac = -0.5*dt/(2*h), ffac = 0.5*dt;

  #pragma omp parallel
  {
    VectorLab vlab; vlab.prepare(*(sim.vel ), stenBeg, stenEnd, 0);
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      vlab.load( velInfo[i], 0); // loads vel field with ghosts
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab& __restrict__ P = plab;
      const VectorLab& __restrict__ V = vlab;
            ScalarBlock&__restrict__ PRHS =*(ScalarBlock*) pRHSInfo[i].ptrBlock;
            VectorBlock&__restrict__ TMPV =*(VectorBlock*) tmpVInfo[i].ptrBlock;
      const VectorBlock&__restrict__    F =*(VectorBlock*)forceInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dU = dU_adv_dif(V, UINF, afac, dfac, ix, iy);
        const Real dV = dV_adv_dif(V, UINF, afac, dfac, ix, iy);

        //const Real gravFac = dt * (1 - o(ix,iy).invRho);
        //const Real duGrav = g[0] * gravFac, dvGrav = g[1] * gravFac;
        const Real duPres = pfac*(P(ix+1,iy).s - P(ix-1,iy).s);
        const Real dvPres = pfac*(P(ix,iy+1).s - P(ix,iy-1).s);

        TMPV(ix,iy).u[0] = V(ix,iy).u[0] +duPres +dU +ffac*F(ix,iy).u[0];
        TMPV(ix,iy).u[1] = V(ix,iy).u[1] +dvPres +dV +ffac*F(ix,iy).u[1];
        PRHS(ix,iy).s += divVel(V, divFac, ix, iy);
      }
    }
  }
}

void advDiff_RK::step2(const double dt)
{
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

      #ifdef DIV_ADVECT
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
          pRHS(ix,0).s -= divFac * dV_adv_dif(TMP,UINF,afac,dfac,ix,-1);
      #endif

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        #ifdef DIV_ADVECT
          pRHS(0,iy).s -= divFac * dU_adv_dif(TMP,UINF,afac,dfac,-1,iy);
        #endif

        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          const Real dU = dU_adv_dif(TMP, UINF, afac, dfac, ix, iy);
          const Real dV = dV_adv_dif(TMP, UINF, afac, dfac, ix, iy);

          vel(ix,iy).u[0] = vel(ix,iy).u[0] + dt * dU;
          vel(ix,iy).u[1] = vel(ix,iy).u[1] + dt * dV;
          #ifdef DIV_ADVECT
            if(ix>      0) pRHS(ix-1,iy  ).s += divFac * dU;
            if(iy>      0) pRHS(ix  ,iy-1).s += divFac * dV;
            if(ix<sizeX-1) pRHS(ix+1,iy  ).s -= divFac * dU;
            if(iy<sizeY-1) pRHS(ix  ,iy+1).s -= divFac * dV;
          #endif
        }

        #ifdef DIV_ADVECT
         pRHS(sizeX-1,iy).s += divFac * dU_adv_dif(TMP,UINF,afac,dfac,sizeX,iy);
        #endif
      }

      #ifdef DIV_ADVECT
       for(int ix=0; ix<VectorBlock::sizeX; ++ix)
         pRHS(ix,sizeY-1).s += divFac * dV_adv_dif(TMP,UINF,afac,dfac,ix,sizeY);
      #endif
    }
  }
}

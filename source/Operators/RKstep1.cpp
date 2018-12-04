//
//  OperatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "RKstep1.h"

void RKstep1::operator()(const double dt)
{
  sim.startProfiler("RKstep1");
  static constexpr int stenBeg[3] = {-1, -1, 0};
  static constexpr int stenEnd[3] = { 2,  2, 1};
  const Real U[]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};

  const Real dfac = (sim.nu/h) * (.5*dt/h), afac = - 0.5 * dt / (2*h);
  const Real divFac = 1.0 / (2*h) / dt,     pfac = - 0.5 * dt / (2*h);

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
      ScalarBlock & __restrict__ PRHS = *(ScalarBlock*)pRHSInfo[i].ptrBlock;
      VectorBlock & __restrict__ TMPV = *(VectorBlock*)tmpVInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real upx = V(ix+1,iy  ).u[0], vpx = V(ix+1,iy  ).u[1];
        const Real upy = V(ix  ,iy+1).u[0], vpy = V(ix  ,iy+1).u[1];
        const Real ucc = V(ix  ,iy  ).u[0], vcc = V(ix  ,iy  ).u[1];
        const Real ulx = V(ix-1,iy  ).u[0], vlx = V(ix-1,iy  ).u[1];
        const Real uly = V(ix  ,iy-1).u[0], vly = V(ix  ,iy-1).u[1];

        //const Real gravFac = dt * (1 - o(ix,iy).invRho);
        //const Real duGrav = g[0] * gravFac, dvGrav = g[1] * gravFac;
        const Real duDiff = dfac*(upx + upy + ulx + uly - 4 * ucc);
        const Real dvDiff = dfac*(vpx + vpy + vlx + vly - 4 * vcc);
        const Real duAdvc = afac*((ucc+U[0])*(upx-ulx) +(vcc+U[1])*(upy-uly));
        const Real dvAdvc = afac*((ucc+U[0])*(vpx-vlx) +(vcc+U[1])*(vpy-vly));
        const Real duPres = pfac*(P(ix+1, iy  ).s - P(ix-1, iy  ).s);
        const Real dvPres = pfac*(P(ix  , iy+1).s - P(ix  , iy-1).s);

        TMPV(ix,iy).u[0] += ucc + duPres + duDiff + duAdvc;
        TMPV(ix,iy).u[1] += vcc + dvPres + dvDiff + dvAdvc;
        PRHS(ix,iy).s += divFac * ((upx - ulx) + (vpy - vly));
      }
    }
  }
  sim.stopProfiler();
}

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


// This operator assumes that obects have put signed distance on the grid

#include "presRHS_step1.h"

// computes: - \chi_{t+1} div( Udef_{t+1})
void presRHS_step1::operator()(const double dt)
{
  sim.startProfiler("presRHS_step1");
  const std::vector<BlockInfo>& pRHSInfo = sim.pRHS->getBlocksInfo();
  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = {2,2,1};
  const Real divFac = 0.5 * sim.getH() / dt;

  #pragma omp parallel
  {
    VectorLab udeflab; udeflab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      udeflab.load(uDefInfo[i], 0); // loads def velocity field with ghosts
      const   VectorLab& __restrict__ UDEF = udeflab;
            ScalarBlock& __restrict__ RHS = *(ScalarBlock*)pRHSInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++) {
        const Real upx = UDEF(ix+1, iy).u[0], vpy = UDEF(ix, iy+1).u[1];
        const Real ulx = UDEF(ix-1, iy).u[0], vly = UDEF(ix, iy-1).u[1];
        RHS(ix,iy).s = - divFac * CHI(ix,iy).s * ((upx-ulx)+(vpy-vly));
      }
    }
  }
  sim.stopProfiler();
}

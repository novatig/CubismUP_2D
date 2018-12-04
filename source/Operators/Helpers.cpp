//
//  ProcessOperatorsOMP.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "Helpers.h"

Real findMaxU::run() const
{
  const Real UINF = sim.uinfx, VINF = sim.uinfy;
  Real U=0, V=0, u=0, v=0;
  #pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      const Real tU = std::fabs(VEL(ix,iy).u[0] + UINF);
      const Real tV = std::fabs(VEL(ix,iy).u[1] + VINF);
      const Real tu = std::fabs(VEL(ix,iy).u[0]);
      const Real tv = std::fabs(VEL(ix,iy).u[1]);
      U = std::max( U, tU );
      V = std::max( V, tV );
      u = std::max( u, tu );
      v = std::max( v, tv );
    }
  }

  return std::max( { U, V, u, v } );
}

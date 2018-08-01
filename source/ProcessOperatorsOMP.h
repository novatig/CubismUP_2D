//
//  ProcessOperatorsOMP.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "Definitions.h"
#include "Shape.h"
#include "GenericOperator.h"

template<typename Lab>
inline double findMaxA(const SimulationData& sim)
{
	Real maxA = 0;
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();

	const int stencil_start[3] = {-1,-1, 0};
	const int stencil_end[3]   = { 2, 2, 1};

 	#pragma omp parallel
	{
		Lab lab;
		lab.prepare(*(sim.grid), stencil_start, stencil_end, true);

		#pragma omp for schedule(static) reduction(max:maxA)
		for(size_t i=0; i<vInfo.size(); i++) {
			lab.load(vInfo[i], 0);

			const BlockInfo info = vInfo[i];
			const FluidBlock& b = *(FluidBlock*)info.ptrBlock;
			const Real inv2h = info.h_gridpoint;

			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
				for(int ix=0; ix<FluidBlock::sizeX; ++ix)
				{
					Real dudx = std::fabs((lab(ix+1,iy  ).u-lab(ix-1,iy  ).u) * inv2h );
					Real dudy = std::fabs((lab(ix  ,iy+1).u-lab(ix  ,iy-1).u) * inv2h );
					Real dvdx = std::fabs((lab(ix+1,iy  ).v-lab(ix-1,iy  ).v) * inv2h );
					Real dvdy = std::fabs((lab(ix  ,iy+1).v-lab(ix  ,iy-1).v) * inv2h );
					maxA = std::max(std::max(dudx,dudy),std::max(dvdx,dvdy));
				}
		}
	}

	return maxA;
}

// -gradp, divergence, advection
template<typename Lab, typename Kernel>
inline void processOMP(const SimulationData& sim)
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
	#pragma omp parallel
	{
		Kernel kernel(sim.dt);
		Lab mylab;
		mylab.prepare(sim.grid, kernel.stencil_start, kernel.stencil_end, true);

		#pragma omp for schedule(static)
		for(size_t i=0; i<vInfo.size(); i++) {
			mylab.load(vInfo[i], 0);
			kernel(mylab, vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
		}
	}
}

inline double findMaxUOMP(const SimulationData& sim)
{
  Real U=0, V=0, u=0, v=0;
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  #pragma omp parallel for schedule(static) reduction(max:U, V, u, v)
  for(size_t i=0; i<vInfo.size(); i++) {
    const BlockInfo& info = vInfo[i];
    FluidBlock& b = *(FluidBlock*)info.ptrBlock;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real tU = std::fabs(b(ix,iy).u + sim.uinfx);
        const Real tV = std::fabs(b(ix,iy).v + sim.uinfy);
        const Real tu = std::fabs(b(ix,iy).u), tv = std::fabs(b(ix,iy).v);
        U = std::max( U, tU );
        V = std::max( V, tV );
        u = std::max( u, tu );
        v = std::max( v, tv );
    }
  }

  return std::max( { U, V, u, v } );
}

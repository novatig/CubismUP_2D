//
//  ProcessOperatorsOMP.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_ProcessOperators_h
#define CubismUP_2D_ProcessOperators_h

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
	Real maxU = 0;
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
	#pragma omp parallel for schedule(static) reduction(max:maxU)
	for(size_t i=0; i<vInfo.size(); i++) {
		const BlockInfo info = vInfo[i];
		FluidBlock& b = *(FluidBlock*)info.ptrBlock;

		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
				maxU = std::max( maxU, std::fabs( b(ix,iy).u + sim.uinfx ) );
				maxU = std::max( maxU, std::fabs( b(ix,iy).v + sim.uinfy ) );
				maxU = std::max( maxU, std::fabs( b(ix,iy).u ) );
				maxU = std::max( maxU, std::fabs( b(ix,iy).v ) );
			}
	}

	return maxU;
}

/*
void computeForcesFromVorticity(vector<BlockInfo>& myInfo, FluidGrid & grid,
	Real ub[2], Real oldAccVort[2], Real rhoS)
{
	Real mU = 0;
	Real mV = 0;
	Real mass = 0;
	const int N = myInfo.size();

	#pragma omp parallel for schedule(static) reduction(+:mU,mV,mass)
	for(int i=0; i<N; i++)
	{
		BlockInfo info = myInfo[i];
		FluidBlock& b = *(FluidBlock*)info.ptrBlock;

		Real h = info.h_gridpoint;

		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				Real p[2];
				info.pos(p, ix, iy);

				mU += - (1-b(ix,iy).chi) * p[1] * b(ix,iy).tmp * b(ix,iy).rho;
				mV -= - (1-b(ix,iy).chi) * p[0] * b(ix,iy).tmp * b(ix,iy).rho;
				mass += b(ix,iy).chi * rhoS;
			}
	}

	ub[0] += (mU-oldAccVort[0]) / mass;
	ub[1] += (mV-oldAccVort[1]) / mass;
	oldAccVort[0] = mU;
	oldAccVort[1] = mV;
}
*/
/*
class OperatorVorticityTmp : public GenericLabOperator
{
 public:
	OperatorVorticityTmp(const double dt)
	{
		stencil_start[0] = -1;
		stencil_start[1] = -1;
		stencil_start[2] = 0;
		stencil_end[0] = 2;
		stencil_end[1] = 2;
		stencil_end[2] = 1;
	}
	~OperatorVorticityTmp() {}

	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
	{
		const Real factor = 0.5/info.h_gridpoint;

		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				const Real vW = lab(ix-1,iy).v;
				const Real vE = lab(ix+1,iy).v;
				const Real uS = lab(ix,iy-1).u;
				const Real uN = lab(ix,iy+1).u;
				o(ix,iy).tmp  = factor * ((vE-vW) - (uN-uS));
			}
	}
};
*/
/*
class OperatorVorticity : public GenericLabOperator
{
 private:
	Layer & vorticity;

 public:
	OperatorVorticity(Layer & vorticity) : vorticity(vorticity)
	{
		stencil_start[0] = -1;
		stencil_start[1] = -1;
		stencil_start[2] = 0;
		stencil_end[0] = 2;
		stencil_end[1] = 2;
		stencil_end[2] = 1;
	}

	~OperatorVorticity() {}

	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
	{
		const Real factor = 0.5/info.h_gridpoint;
		const int bx = info.index[0]*FluidBlock::sizeX;
		const int by = info.index[1]*FluidBlock::sizeY;

		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				const Real vW = lab(ix-1,iy).v;
				const Real vE = lab(ix+1,iy).v;
				const Real uS = lab(ix,iy-1).u;
				const Real uN = lab(ix,iy+1).u;

				o(ix,iy).tmp                = factor * ((vE-vW) - (uN-uS));
				vorticity(bx + ix, by + iy) = factor * ((vE-vW) - (uN-uS));
			}
	}
};

// divergence with layer - still useful for diagnostics
template<typename Lab, typename Kernel>
void processOMP(Layer& outputField, vector<BlockInfo>& myInfo, FluidGrid & grid)
{
	BlockInfo * ary = &myInfo.front();
	const int N = myInfo.size();

	#pragma omp parallel
	{
		Kernel kernel(outputField);

		Lab mylab;
		mylab.prepare(grid, kernel.stencil_start, kernel.stencil_end, true);

		#pragma omp for schedule(static)
		for (int i=0; i<N; i++)
		{
			mylab.load(ary[i], 0, false);

			kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
		}
	}
}

// divergence split with layer - still useful for diagnostics
template<typename Lab, typename Kernel>
void processOMP(Layer& outputField, const Real rho0, const Real dt, c
	onst int step, vector<BlockInfo>& myInfo, FluidGrid & grid)
{
	BlockInfo * ary = &myInfo.front();
	const int N = myInfo.size();

	#pragma omp parallel
	{
		Kernel kernel(outputField, rho0, dt, step);

		Lab mylab;
		mylab.prepare(grid, kernel.stencil_start, kernel.stencil_end, true);

		#pragma omp for schedule(static)
		for (int i=0; i<N; i++)
		{
			mylab.load(ary[i], 0, false);

			kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
		}
	}
}
*/
#endif

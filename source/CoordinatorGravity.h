//
//  CoordinatorGravity.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorGravity_h
#define CubismUP_2D_CoordinatorGravity_h

#include "GenericCoordinator.h"
#include "GenericOperator.h"

class OperatorGravity : public GenericOperator
{
 private:
	const double dt;
	const Real g[2];

 public:
	OperatorGravity(Real g[2], double dt) : dt(dt), g{g[0],g[1]} {}
	~OperatorGravity() {}

	void operator()(const BlockInfo& info, FluidBlock& block) const
	{
		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				Real p[2];
				info.pos(p, ix, iy);

				block(ix,iy).u += dt * g[0];
				block(ix,iy).v += dt * g[1];
			}
	}
};

class CoordinatorGravity : public GenericCoordinator
{
 protected:
	Real gravity[2];

 public:
	CoordinatorGravity(Real gravity[2], FluidGrid * grid) :
	GenericCoordinator(grid), gravity{gravity[0],gravity[1]}
	{
	}

	void operator()(const double dt)
	{
		check("gravity - start");

		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();

		#pragma omp parallel
		{
			OperatorGravity kernel(gravity, dt);

			#pragma omp for schedule(static)
			for (int i=0; i<N; i++)
				kernel(ary[i], *(FluidBlock*)ary[i].ptrBlock);
		}

		check("gravity - end");
	}

	string getName()
	{
		return "Gravity";
	}
};


#endif

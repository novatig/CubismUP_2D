//
//  CoordinatorComputeShape.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorComputeShape_h
#define CubismUP_2D_CoordinatorComputeShape_h

#include "GenericCoordinator.h"
#include "Shape.h"

class CoordinatorComputeShape : public GenericCoordinator
{
 protected:
	const Real*const uBody;
  const Real*const vBody;
  const Real*const omegaBody;
	Shape*const shape;

 public:
	CoordinatorComputeShape(const Real*const u, const Real*const v, const Real*const w, Shape*const s, FluidGrid*const g) :
  GenericCoordinator(g), uBody(u), vBody(v), omegaBody(w), shape(s) { }

	void operator()(const double dt)
	{
		check("shape - start");

		const Real ub[2] = { *uBody, *vBody };
		shape->updatePosition(ub, *omegaBody, dt);

		const Real domainSize[2] = {
			grid->getBlocksPerDimension(0)*FluidBlock::sizeX*vInfo[0].h_gridpoint,
			grid->getBlocksPerDimension(1)*FluidBlock::sizeY*vInfo[0].h_gridpoint
		};
		Real p[2] = {0,0};
		shape->getCentroid(p);

		if (p[0]<0 || p[0]>domainSize[0] || p[1]<0 || p[1]>domainSize[1]) {
			cout << "Body out of domain: " << p[0] << " " << p[1] << endl;
			exit(0);
		}

		#pragma omp parallel for schedule(static)
		for(int i=0; i<vInfo.size(); i++) {
			FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;

			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				b(ix,iy).rho = 1;
				b(ix,iy).tmp = 0;
			}
		}

		shape->create(vInfo);

		check("shape - end");
	}

	string getName()
	{
		return "ComputeShape";
	}
};

#endif

//
//  CoordinatorPenalization.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorPenalization_h
#define CubismUP_2D_CoordinatorPenalization_h

#include "GenericCoordinator.h"
#include "Shape.h"

class OperatorPenalization : public GenericOperator
{
 private:
  const double dt;
	const Real uBody[2],omegaBody,centerOfMass[2];
	const double lambda;

 public:
	OperatorPenalization(double dt, Real uSolid, Real vSolid, Real omegaBody, Real xCenterOfMass, Real yCenterOfMass, double lambda) :
	dt(dt), uBody{uSolid,vSolid}, omegaBody(omegaBody), centerOfMass{xCenterOfMass,yCenterOfMass}, lambda(lambda) {}
  ~OperatorPenalization() {}

  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
		{
			Real p[2] = {0,0};
      info.pos(p,ix,iy);
      p[0] -= centerOfMass[0];
      p[1] -= centerOfMass[1];
      const Real alpha = 1./(1. + dt * lambda * block(ix,iy).chi);

			#ifndef _MOVING_FRAME_
	      const Real U_TOT[2] = {
	      		uBody[0] - omegaBody*p[1],
						uBody[1] + omegaBody*p[0]
				};
			#else
	      const Real U_TOT[2] = {
	      		- omegaBody*p[1],
						+ omegaBody*p[0]
				};
			#endif
      block(ix,iy).u = alpha*block(ix,iy).u + (1-alpha)*U_TOT[0];
      block(ix,iy).v = alpha*block(ix,iy).v + (1-alpha)*U_TOT[1];
		}
  }
};

class CoordinatorPenalization : public GenericCoordinator
{
 protected:
	Real *uBody, *vBody, *omegaBody;
	Shape * shape;
	Real * lambda;

 public:
	CoordinatorPenalization(Real * uBody, Real * vBody, Real * omegaBody, Shape * shape, Real * lambda, FluidGrid * grid) :
		GenericCoordinator(grid), uBody(uBody), vBody(vBody), omegaBody(omegaBody), shape(shape), lambda(lambda)
	{
	}

	void operator()(const double dt)
	{
		check("penalization - start");

		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();

		#pragma omp parallel
		{
			Real com[2];
			shape->getCenterOfMass(com);
			OperatorPenalization kernel(dt, *uBody, *vBody, *omegaBody, com[0], com[1], *lambda);

			#pragma omp for schedule(static)
			for(int i=0; i<N; i++)
				kernel(ary[i], *(FluidBlock*)ary[i].ptrBlock);
		}

		check("penalization - end");
	}

	string getName()
	{
		return "Penalization";
	}
};

#endif

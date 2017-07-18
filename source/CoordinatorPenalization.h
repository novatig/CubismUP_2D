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

class CoordinatorPenalization : public GenericCoordinator
{
 protected:

  const Real* const uBody;
  const Real* const vBody;
  const Real* const omegaBody;
  const Real* const lambda;
	const Shape* const shape;

 public:
	CoordinatorPenalization(Real*uBody, Real*vBody, Real*omegaBody, Shape*shape, Real*lambda, FluidGrid*grid) :
		GenericCoordinator(grid), uBody(uBody), vBody(vBody), omegaBody(omegaBody), shape(shape), lambda(lambda)
	{
	}

	void operator()(const double dt)
	{
		check("penalization - start");

		shape->penalize(*uBody, *vBody, *omegaBody, dt, *lambda, vInfo);

		check("penalization - end");
	}

	string getName()
	{
		return "Penalization";
	}
};

#endif

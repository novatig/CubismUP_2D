//
//  CoordinatorBodyVelocities.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorComputeBodyVelocities_h
#define CubismUP_2D_CoordinatorComputeBodyVelocities_h

class CoordinatorBodyVelocities : public GenericCoordinator
{
protected:
	Real* const uBody;
	Real* const vBody;
	Real* const omegaBody;
	const Real* const lambda;
	Shape* const shape;

public:
	CoordinatorBodyVelocities(Real*const u, Real*const v, Real*const w, Shape*const s, Real*const l, FluidGrid*const g) : GenericCoordinator(g),uBody(u),vBody(v),omegaBody(w),lambda(l),shape(s)	{ }

	void operator()(const double dt)
	{
		shape->computeVelocities(uBody, vBody, omegaBody, vInfo);

		#ifdef RL_MPI_CLIENT
		shape->act(uBody, vBody, omegaBody, dt);
		#endif
	}

	string getName()
	{
		return "BodyVelocities";
	}
};
#endif

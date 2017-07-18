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

	#ifdef RL_MPI_CLIENT
		Communicator*const communicator;
		Real*const time;
	#endif

public:
	CoordinatorBodyVelocities(Real*const u, Real*const v, Real*const w, Real*const s, Real*const l, FluidGrid*const g
		#ifdef RL_MPI_CLIENT
		, Communicator*const c, Real*const t
		#endif
	) : GenericCoordinator(g),uBody(u),vBody(v),omegaBody(w),lambda(l),shape(s)
		#ifdef RL_MPI_CLIENT
		, communicator(c), time(t)
		#endif
	{ }

	void operator()(const double dt)
	{
		computeVelocities(uBody, vBody, omegaBody, vInfo);

		#ifdef RL_MPI_CLIENT
		shape->act(uBody, vBody, omegaBody, time, communicator)
		#endif
	}

	string getName()
	{
		return "BodyVelocities";
	}
};
#endif

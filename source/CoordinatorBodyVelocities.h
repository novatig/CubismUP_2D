//
//  CoordinatorBodyVelocities.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorComputeBodyVelocities_h
#define CubismUP_2D_CoordinatorComputeBodyVelocities_h

#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10

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
		_AGENT_STATUS info = _AGENT_FIRSTCOMM;
		unsigned iter = 0;
		bool initialized_time_next_comm = false;
		Real Torque = 0, time_next_comm = 0, old_Torque = 0, old_Dist = 100;
		Real powerOutput = 0, old_powerOutput = 0;
		Real*const time;
	#endif

public:
	CoordinatorBodyVelocities(Real * uBody, Real * vBody, Real * omegaBody,
		Shape * shape, Real * lambda, FluidGrid * grid
		#ifdef RL_MPI_CLIENT
		, Communicator*const comm, Real * t
		#endif
	) :
		GenericCoordinator(grid), uBody(uBody), vBody(vBody), omegaBody(omegaBody),
		lambda(lambda), shape(shape)
		#ifdef RL_MPI_CLIENT
		, communicator(comm), time(t)
		#endif
	{
	}

	void operator()(const double dt)
	{
		Real centerOfMass[2];
		shape->getCenterOfMass(centerOfMass);

		// gravity acts on the center of gravity (for constant g: center of mass)
		// buoyancy acts on the center of buoyancy (for fully immersed body: center of geometry/centroid)
		double mass = 0;
		double u = 0;
		double v = 0;
		double momOfInertia = 0;
		double angularMomentum = 0;
		const int N = vInfo.size();


		#pragma omp parallel for schedule(static) reduction(+:u,v,momOfInertia,angularMomentum,mass)
		for(int i=0; i<N; i++)
		{
			BlockInfo info = vInfo[i];
			FluidBlock& b = *(FluidBlock*)info.ptrBlock;

			double h = info.h_gridpoint;

			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
				for(int ix=0; ix<FluidBlock::sizeX; ++ix)
				{
					double p[2] = {0,0};
					info.pos(p, ix, iy);
					p[0] -= centerOfMass[0];
					p[1] -= centerOfMass[1];
					const double rhochi = b(ix,iy).rho * b(ix,iy).chi;
					mass += rhochi;
					u += b(ix,iy).u * rhochi;
					v += b(ix,iy).v * rhochi;
					momOfInertia   +=rhochi*(p[0]*p[0]       + p[1]*p[1]);
					angularMomentum+=rhochi*(p[0]*b(ix,iy).v - p[1]*b(ix,iy).u);
				}
		}

		*uBody = u / mass;
		*vBody = v / mass;

		*omegaBody = angularMomentum / momOfInertia;
		shape->M = mass * vInfo[0].h_gridpoint * vInfo[0].h_gridpoint;
		shape->J = momOfInertia * vInfo[0].h_gridpoint * vInfo[0].h_gridpoint;

		#ifdef RL_MPI_CLIENT
		if(!initialized_time_next_comm || *time>time_next_comm)
		{
			initialized_time_next_comm = true;
			time_next_comm = time_next_comm + 0.5;
			const Real rhoS = shape->rhoS;
			const Real angle = shape->getOrientation(), omega = *omegaBody;
			const Real cosTheta = std::cos(angle), sinTheta = std::sin(angle);
			const Real a=max(shape->semiAxis[0], shape->semiAxis[1]);
			const Real b=min(shape->semiAxis[0], shape->semiAxis[1]);
			//Characteristic scales:
			const Real lengthscale = a;
			const Real velscale = std::sqrt((rhoS/1.-1)*9.8*b);
			const Real torquescale = 1/8 *M_PI*1*pow((a*a-b*b)*velscale,2)*a/b;
			//Nondimensionalization:
			const Real xdot=*uBody/velscale,ydot=*vBody/velscale,T=Torque/torquescale;
			const Real X =shape->labCenterOfMass[0]/a, Y =shape->labCenterOfMass[1]/a;
			const Real U =xdot*cosTheta+ydot*sinTheta, V =ydot*cosTheta-xdot*sinTheta;

			const bool ended = X>125 || X<-10 || Y<=-50;
			const bool landing = std::fabs(angle - .25*M_PI) < 0.1;
			const Real vertDist = std::fabs(Y+50), horzDist = std::fabs(X-100);

			Real reward;
			if (ended) {
				info = _AGENT_LASTCOMM;
				reward= (X>125 || X<-10) ? -100 -HEIGHT_PENAL*vertDist
							: (horzDist<1? (landing?2:1) * TERM_REW_FAC : -horzDist) ;
			} else
			reward = (old_Dist-horzDist) -fabs(Torque-old_Torque)/0.5; //-(powerOutput-old_powerOutput);

			vector<double> state = {U,V,omega,X,Y,cosTheta,sinTheta,T,xdot,ydot}; vector<double> action = {0.};
			communicator->sendState(0, info, state, reward);

			if(info == _AGENT_LASTCOMM) abort();
			old_Dist = horzDist;
			old_Torque = Torque;
			old_powerOutput = powerOutput;
			info = _AGENT_NORMALCOMM;

			communicator->recvAction(action);
			Torque = action[0]*torquescale;
		}
		*omegaBody += dt*Torque/shape->J;
		powerOutput += dt*Torque*Torque;
		#endif
	}

	string getName()
	{
		return "BodyVelocities";
	}
};


#endif

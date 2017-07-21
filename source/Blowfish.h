//
//  Shape.h
//  CubismUP_2D
//
//	Virtual shape class which defines the interface
//	Default simple geometries are also provided and can be used as references
//
//	This class only contains static information (position, orientation,...), no dynamics are included (e.g. velocities,...)
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_Blowfish_h
#define CubismUP_2D_Blowfish_h
#include "Shape.h"

class Blowfish : public Shape
{
 protected:
	const Real radius;
	const Real rhoTop = 1.5; //top half
	const Real rhoBot = 0.5; //bot half
	const Real rhoFin = 1.0; //fins

	const Real finLength = 0.5*radius; //fins
	const Real finWidth  = 0.1*radius; //fins
	const Real finAngle0 = M_PI/6; //fins

	const Real attachDist = radius+finWidth;
	Real flapAng_R = 0, flapAng_L = 0;
	Real flapVel_R = 0, flapVel_L = 0;
	Real flapAcc_R = 0, flapAcc_L = 0;
	//Real powerOutput = 0, old_powerOutput = 0;
	const Real rhoF = 1; //ASSUME RHO FLUID == 1
	const Real lengthscale = 2*radius;
	const Real distHalfCM = 4*radius/(3*M_PI);
	const Real halfarea = 0.5*M_PI*radius*radius;
	const Real minRho = min(rhoTop,rhoBot), maxRho = max(rhoTop,rhoBot);
	const Real forceDw = halfarea*(rhoF-minRho)*9.8;
	const Real forceUp = halfarea*(maxRho-rhoF)*9.8;
	const Real torquescale = distHalfCM*(forceDw+forceUp);
	const Real velscale = sqrt(torquescale/lengthscale/lengthscale/rhoF);
	const Real timescale = lengthscale/velscale;

 public:
	Blowfish(Real C[2],const Real ang,const Real R):Shape(C,ang,0.5),radius(R)
	{
		// based on weighted average
		const Real CentTop =  distHalfCM;
		const Real MassTop =  halfarea*rhoTop;
		const Real CentBot = -distHalfCM;
		const Real MassBot =  halfarea*rhoBot;
		const Real CentFin = -sin(finAngle0)*(attachDist+finLength/2);
		const Real MassFin = 2*finLength*finWidth*rhoFin;

		d_gm[0] = 0;
		d_gm[1] = -(CentTop*MassTop + CentBot*MassBot + CentFin*MassFin)/(MassTop + MassBot + MassFin);

		centerOfMass[0]=center[0]-cos(orientation)*d_gm[0]+sin(orientation)*d_gm[1];
		centerOfMass[1]=center[1]-sin(orientation)*d_gm[0]-cos(orientation)*d_gm[1];
	}

	#ifdef RL_MPI_CLIENT
		void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) override
		{
			assert(time_ptr not_eq nullptr);
			assert(communicator not_eq nullptr);
			if(!initialized_time_next_comm || *time_ptr>time_next_comm)
			{
				initialized_time_next_comm = true;
				time_next_comm = time_next_comm + 0.1*timescale;
				//compute torque scale:

				const Real w = (*omegaBody)*timescale, angle = orientation;
				const Real u = (*uBody)/velscale, v = (*vBody)/velscale;
				const Real cosAng = cos(angle), sinAng = sin(angle);
				const Real U = u*cosAng + v*sinAng, V = v*cosAng - u*sinAng;
				const Real WR = flapVel_R*timescale, WL = flapVel_L*timescale;

				const bool ended = cosAng<0; //(angle>M_PI || angle<-M_PI);
				const Real reward = ended ? -10 : cosAng -sqrt(u*u+v*v);
				if (ended) {
					printf("End of episode due to angle: %f\n", orientation);
					info = _AGENT_LASTCOMM;
				}

				vector<double> states = {U, V, w, angle, flapAng_R, flapAng_L, WR, WL};
				vector<double> action = {0, 0};

		    printf("(%u) Sending (%lu) [%f %f %f %f %f %f %f %f]\n",
					iter++, states.size(), U, V, w, angle, flapAng_R, flapAng_L, WR, WL);

				communicator->sendState(0, info, states, reward);

				if(info == _AGENT_LASTCOMM) throw iter;
				else info = _AGENT_NORMALCOMM;

				communicator->recvAction(action);
		    printf("Received %f %f\n", action[0], action[1]);
				flapAcc_R = action[0]/timescale/timescale;
				flapAcc_L = action[1]/timescale/timescale;
			}
		}
	#else
		void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) override
		{
			*omegaBody = 0;
			const Real omega = 2*M_PI/timescale;
			const Real amp = omega*omega*M_PI/8;
			printf("V:%f, L:%f, Flap amp %f, omega %f\n", velscale, lengthscale, amp, omega);
			//accelation is a sine, therefore angvel is cosine and angle is sine
			flapAcc_R =  amp*std::sin(omega*(*time_ptr));
			flapAcc_L = -amp*std::sin(omega*(*time_ptr));
		}
	#endif

	void updatePosition(const Real u[2], Real omega, Real dt) override
	{
		Shape::updatePosition(u, omega, dt);
		if(flapAng_R > M_PI/2) { //maximum extent of fin is pi/2
			printf("Blocked flapAng_R at  M_PI/2\n");
			flapAng_R =  M_PI/2;
			if(flapVel_R>0) flapVel_R = 0;
			if(flapAcc_R>0) flapAcc_R = 0;
		}
		if(flapAng_R < -M_PI/2) { //maximum extent of fin is pi/2
			printf("Blocked flapAng_R at -M_PI/2\n");
			flapAng_R = -M_PI/2;
			if(flapVel_R<0) flapVel_R = 0;
			if(flapAcc_R<0) flapAcc_R = 0;
		}
		if(flapAng_L > M_PI/2) { //maximum extent of fin is pi/2
			printf("Blocked flapAng_L at  M_PI/2\n");
			flapAng_L =  M_PI/2;
			if(flapVel_L>0) flapVel_L = 0;
			if(flapAcc_L>0) flapAcc_L = 0;
		}
		if(flapAng_L < -M_PI/2) { //maximum extent of fin is pi/2
			printf("Blocked flapAng_L at -M_PI/2\n");
			flapAng_L = -M_PI/2;
			if(flapVel_L<0) flapVel_L = 0;
			if(flapAcc_L<0) flapAcc_L = 0;
		}

		flapAng_R += dt*flapVel_R + .5*dt*dt*flapAcc_R;
		flapAng_L += dt*flapVel_L + .5*dt*dt*flapAcc_L;
		flapVel_R += dt*flapAcc_R;
		flapVel_L += dt*flapAcc_L;
		printf("[ang, angvel, angacc] : right:[%f %f %f] left:[%f %f %f]\n",
		 flapAng_R, flapVel_R, flapAcc_R, flapAng_L, flapVel_L, flapAcc_L);
	}

	void create(const vector<BlockInfo>& vInfo) override
	{
		const Real angleAttFin1 = orientation  -finAngle0;
		const Real angleAttFin2 = orientation  +finAngle0 +M_PI;
		const Real angleTotFin1 = angleAttFin1 +flapAng_R;
		const Real angleTotFin2 = angleAttFin2 +flapAng_L;

		const Real attach1[2] = {
			center[0]+attachDist*cos(angleAttFin1),
			center[1]+attachDist*sin(angleAttFin1)
		};
		const Real attach2[2] = {
			center[0]+attachDist*cos(angleAttFin2),
			center[1]+attachDist*sin(angleAttFin2)
		};

		const Real h = vInfo[0].h_gridpoint;
		for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

		const FillBlocks_Cylinder kernelC(radius, h, center, rhoBot);
		const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, angleTotFin1, flapVel_R, rhoFin);
		const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, angleTotFin2, flapVel_L, rhoFin);

    for(int i=0; i<vInfo.size(); i++) {
    	BlockInfo info = vInfo[i];
  		if(kernelC._is_touching(info))
			{
  			assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
  			obstacleBlocks[info.blockID] = new ObstacleBlock;
  			obstacleBlocks[info.blockID]->clear(); //memset 0
  		}
			else if(kernelF1._is_touching(info))
			{
				assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
  			obstacleBlocks[info.blockID] = new ObstacleBlock;
  			obstacleBlocks[info.blockID]->clear(); //memset 0
			}
			else if(kernelF2._is_touching(info))
			{
				assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
  			obstacleBlocks[info.blockID] = new ObstacleBlock;
  			obstacleBlocks[info.blockID]->clear(); //memset 0
			}
    }

		#pragma omp parallel
		{
			const FillBlocks_VarRhoCylinder kernelC(radius, h, center, rhoTop, rhoBot, orientation);
			const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, angleTotFin1, flapVel_R, rhoFin);
			const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, angleTotFin2, flapVel_L, rhoFin);

	    #pragma omp for schedule(dynamic)
			for(int i=0; i<vInfo.size(); i++) {
				BlockInfo info = vInfo[i];
				const auto pos = obstacleBlocks.find(info.blockID);
				if(pos == obstacleBlocks.end()) continue;
				FluidBlock& b = *(FluidBlock*)info.ptrBlock;
				kernelC(info, b, pos->second);
				kernelF1(info, b, pos->second);
				kernelF2(info, b, pos->second);
			}
		}

		removeMoments(vInfo);
    for (auto & block : obstacleBlocks) block.second->allocate_surface();
	}

	Real getCharLength() const  override
	{
		return 2 * radius;
	}

	void outputSettings(ostream &outStream)
	{
		outStream << "Blowfish\n";
		Shape::outputSettings(outStream);
	}
};

#endif

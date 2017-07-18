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

#ifndef CubismUP_2D_Shape_h
#define CubismUP_2D_Shape_h

class Blowfish : public Shape
{
 protected:
	const Real radius, h;
	const Real rhoTop = 1.5; //top half
	const Real rhoBot = 0.5; //bot half
	const Real rhoFin = 1.0; //fins

	const Real finLength = 0.5*radius; //fins
	const Real finWidth  = 0.1*radius; //fins
	const Real finAngle0 = M_PI/6; //fins

	const Real attachDist = radius +finWidth/2;
	Real flapAng_R = 0, flapAng_L = 0;
	Real flapVel_R = 0, flapVel_L = 0;
	Real flapAcc_R = 0, flapAcc_L = 0;
	//Real powerOutput = 0, old_powerOutput = 0;

	const Real distHalfCM = 4*radius/(3*M_PI);
	const Real halfarea = 0.5*M_PI*radius*radius;
	const Real minRho = min(rhoTop,rhoBot), maxRho = max(rhoTop,rhoBot);
	const Real forceDw = halfarea*(1-minRho)*9.8;
	const Real forceUp = halfarea*(maxRho-1)*9.8;
	const Real torquescale = distHalfCM*(forceDw+forceUp);
	const Real velscale = sqrt(torquescale/radius/radius);
	const Real timescale = radius/velscale;

	void create(const vector<BlockInfo>& vInfo)
	{
		const Real angleAttFin1 = orientation  -finAngle0;
		const Real angleAttFin2 = orientation  +finAngle0 +M_PI;
		const Real angleTotFin1 = angleAttFin1 +flapAng_R;
		const Real angleTotFin2 = angleAttFin2 +flapAng_L;

		const Real attach1[2] = {center[0]+attachDist*cos(angleAttFin1), center[1]+attachDist*sin(angleAttFin1)};
		const Real attach2[2] = {center[0]+attachDist*cos(angleAttFin2), center[1]+attachDist*sin(angleAttFin2)};

		const Real h = vInfo[0].h_gridpoint;
		for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

		const FillBlocks_Cylinder kernelC(radius, h, center, rhoBot);
		const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, rhoFin, angleTotFin1, flapVel_R);
		const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, rhoFin, angleTotFin2, flapVel_L);

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
			const FillBlocks_Cylinder kernelC(radius, h, center, rhoBot);
			const FillBlocks_HalfCylinder kernelH(radius, h, center, rhoTop, orientation);
			const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, rhoFin, angleTotFin1, flapVel_R);
			const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, rhoFin, angleTotFin2, flapVel_L);

	    #pragma omp for
			for(int i=0; i<vInfo.size(); i++) {
				BlockInfo info = vInfo[i];
				const auto pos = obstacleBlocks.find(info.blockID);
				if(pos == obstacleBlocks.end()) continue;
				FluidBlock& b = *(FluidBlock*)info.ptrBlock;
				kernelC(info, b, pos->second);
				kernelH(info, b, pos->second);
				kernelF1(info, b, pos->second);
				kernelF2(info, b, pos->second);
			}
		}

		{
			// Update center of mass
			const Real CentTop =  4*radius/(3*M_PI);
			const Real MassTop = .5*M_PI*radius*radius*rhoTop;
			const Real CentBot = -4*radius/(3*M_PI);
			const Real MassBot = .5*M_PI*radius*radius*rhoBot;
			const Real MassFin = finLength*finWidth*rhoFin;
			const Real totMass = MassTop+MassBot+MassFin*2;
			//angles from frame of reference of obstacle (remove 'orientation')
			const Real cosF1 = cos(-finAngle0+flapAng_R);
			const Real sinF1 = sin(-finAngle0+flapAng_R);
			const Real cosF2 = cos(M_PI+finAngle0+flapAng_L);
			const Real sinF2 = sin(M_PI+finAngle0+flapAng_L);
			const Real cosA1 = cos(-finAngle0), sinA1 = sin(-finAngle0);
			const Real cosA2 = cos(M_PI+finAngle0), sinA2 = sin(M_PI+finAngle0);

			const Real CMxFin1 = cosA1*attachDist + cosF1*(finLength/2);
			const Real CMxFin2 = cosA2*attachDist + cosF2*(finLength/2);
			const Real CMyFin1 = sinA1*attachDist + sinF1*(finLength/2);
			const Real CMyFin2 = sinA2*attachDist + sinF2*(finLength/2);

			d_gm[0] = -(CMxFin1*MassFin + CMxFin2*MassFin)/totMass;
			d_gm[1] = -(CentTop*MassTop + CentBot*MassBot +(CMyFin1+CMyFin2)*MassFin)/totMass;

			centerOfMass[0] = center[0] -cos(orientation)*d_gm[0] +sin(orientation)*d_gm[1];
			centerOfMass[1] = center[1] -sin(orientation)*d_gm[0] -cos(orientation)*d_gm[1];
		}

		removeMoments(vInfo);
	}

 public:
	Blowfish(Real C[2], const Real ang, const Real R): Shape(C,.5,ang), radius(R)
	{
		// based on weighted average
		const Real CentTop =  4*radius/(3*M_PI);
		const Real MassTop = .5*M_PI*radius*radius*rhoTop;
		const Real CentBot = -4*radius/(3*M_PI);
		const Real MassBot = .5*M_PI*radius*radius*rhoBot;
		const Real CentFin = -sin(finAngle0)*(attachDist+finLength/2);
		const Real MassFin = 2*finLength*finWidth*rhoFin;

		d_gm[0] = 0;
		d_gm[1] = -(CentTop*MassTop + CentBot*MassBot + CentFin*MassFin)/(MassTop + MassBot + MassFin);

		centerOfMass[0]=center[0]-cos(orientation)*d_gm[0]+sin(orientation)*d_gm[1];
		centerOfMass[1]=center[1]-sin(orientation)*d_gm[0]-cos(orientation)*d_gm[1];
	}

	#ifdef RL_MPI_CLIENT
	void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real time, const Real dt) override
	{
		if(!initialized_time_next_comm || time>time_next_comm)
		{
			initialized_time_next_comm = true;
			time_next_comm = time_next_comm + 0.1;
			//compute torque scale:


			const Real w = *omegaBody, u = *uBody, v = *vBody;
			const Real cosAng = cos(orientation), sinAng = sin(orientation);

			//Nondimensionalization:
			const Real xdot=u/velscale, ydot=v/velscale, T=Torque/torquescale;
			const Real X = labCenterOfMass[0]/a, Y = labCenterOfMass[1]/a;
			const Real U = xdot*cosAng +ydot*sinAng, V = ydot*cosAng -xdot*sinAng;

			const bool ended = X>125 || X<-10 || Y<=-50;
			const bool landing = std::fabs(angle - .25*M_PI) < 0.1;
			const Real vertDist = std::fabs(Y+50), horzDist = std::fabs(X-100);

			Real reward;
			if (ended)
			{
				info = _AGENT_LASTCOMM;
				reward= (X>125 || X<-10) ? -100 -HEIGHT_PENAL*vertDist
							: (horzDist<1? (landing?2:1) * TERM_REW_FAC : -horzDist) ;
			} else
				reward = (old_Dist-horzDist) -fabs(Torque-old_Torque)/.5;
			//-(powerOutput-old_powerOutput);

			vector<double> state = {U,V,omega,X,Y,cosTheta,sinTheta,T,xdot,ydot}; vector<double> action = {0.};

	    printf("Sending (%lu) [%f %f %f %f %f %f %f %f %f %f], %f %f\n", state.size(),U,V,omega,X,Y,cosTheta,sinTheta,T,xdot,ydot, Torque,torquescale);

			communicator->sendState(0, info, state, reward);

			if(info == _AGENT_LASTCOMM) abort();
			old_Dist = horzDist;
			old_Torque = Torque;
			old_powerOutput = powerOutput;
			info = _AGENT_NORMALCOMM;

			communicator->recvAction(action);
	       printf("Received %f\n", action[0]);
			Torque = action[0]*torquescale;
		}
	}
	#endif

	void updatePosition(const Real u[2], Real omega, Real dt) override
	{
		Shape::updatePosition(u, omega, dt);
		if(flapAng_R > M_PI/2) { //maximum extent of fin is pi/2
			flapAng_R =  M_PI/2;
			if(flapVel_R>0) flapVel_R = 0;
			if(flapAcc_R>0) flapAcc_R = 0;
		}
		if(flapAng_R < -M_PI/2) { //maximum extent of fin is pi/2
			flapAng_R = -M_PI/2;
			if(flapVel_R<0) flapVel_R = 0;
			if(flapAcc_R<0) flapAcc_R = 0;
		}
		if(flapAng_L > M_PI/2) { //maximum extent of fin is pi/2
			flapAng_L =  M_PI/2;
			if(flapVel_L>0) flapVel_L = 0;
			if(flapAcc_L>0) flapAcc_L = 0;
		}
		if(flapAng_L < -M_PI/2) { //maximum extent of fin is pi/2
			flapAng_L = -M_PI/2;
			if(flapVel_L<0) flapVel_L = 0;
			if(flapAcc_L<0) flapAcc_L = 0;
		}

		flapAng_R += dt*flapVel_R + .5*dt*dt*flapAcc_R;
		flapAng_L += dt*flapVel_L + .5*dt*dt*flapAcc_L;
		flapVel_R += dt*flapAcc_R;
		flapVel_L += dt*flapAcc_L;
	}

	Real getCharLength() const
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

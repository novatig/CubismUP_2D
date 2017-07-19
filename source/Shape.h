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

#include "IF2D_ObstacleLibrary.h"
#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10

class Shape
{
 #ifdef RL_MPI_CLIENT
	 public:
			Communicator* communicator = nullptr;
			Real* time_ptr = nullptr;
 #endif

 protected:
	// general quantities
	Real centerOfMass[2], orientation;
	Real center[2]; // for single density, this corresponds to centerOfMass
	Real d_gm[2]; // distance of center of geometry to center of mass
	Real labCenterOfMass[2] = {0,0};
	Real M = 0;
	Real J = 0;

	const Real rhoS;
	std::map<int,ObstacleBlock*> obstacleBlocks;

	#ifdef RL_MPI_CLIENT
		_AGENT_STATUS info = _AGENT_FIRSTCOMM;
		unsigned iter = 0;
		bool initialized_time_next_comm = false;
		Real time_next_comm = 0;
	#endif

	Real smoothHeaviside(Real rR, Real radius, Real eps) const
	{
		if (rR < radius-eps*.5)
			return (Real) 1.;
		else if (rR > radius+eps*.5)
			return (Real) 0.;
		else
			return (Real) ((1.+cos(M_PI*((rR-radius)/eps+.5)))*.5);
	}

	inline void rotate(Real p[2]) const
	{
		const Real x = p[0], y = p[1];
		p[0] =  x*cos(orientation) + y*sin(orientation);
		p[1] = -x*sin(orientation) + y*cos(orientation);
	}

 public:
	Shape(Real center[2], Real orientation, const Real rhoS) :
		center{center[0],center[1]}, centerOfMass{center[0],center[1]}, d_gm{0,0},
    orientation(orientation), rhoS(rhoS)
	{	}

	virtual ~Shape() {}

	virtual Real getCharLength() const = 0;
	virtual void create(const vector<BlockInfo>& vInfo) = 0;

	#ifdef RL_MPI_CLIENT
	virtual void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) {}
	#endif

	virtual void updatePosition(const Real u[2], Real omega, Real dt)
	{
		// update centerOfMass - this is the reference point from which we compute the center
		#ifndef _MOVING_FRAME_
		centerOfMass[0] += dt*u[0];
    centerOfMass[1] += dt*u[1];
		#endif

		labCenterOfMass[0] += dt*u[0];
		labCenterOfMass[1] += dt*u[1];

		orientation += dt*omega;
		orientation = orientation>2*M_PI ? orientation-2*M_PI : (orientation<0 ? orientation+2*M_PI : orientation);

		center[0] = centerOfMass[0] + cos(orientation)*d_gm[0] - sin(orientation)*d_gm[1];
		center[1] = centerOfMass[1] + sin(orientation)*d_gm[0] + cos(orientation)*d_gm[1];
	}

	void setCentroid(Real centroid[2])
	{
		center[0] = centroid[0];
		center[1] = centroid[1];

		centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
		centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
	}

	void setCenterOfMass(Real com[2])
	{
		centerOfMass[0] = com[0];
		centerOfMass[1] = com[1];

		center[0] = centerOfMass[0] + cos(orientation)*d_gm[0] - sin(orientation)*d_gm[1];
		center[1] = centerOfMass[1] + sin(orientation)*d_gm[0] + cos(orientation)*d_gm[1];
	}

	void getCentroid(Real centroid[2]) const
	{
		centroid[0] = center[0];
		centroid[1] = center[1];
	}

	void getCenterOfMass(Real com[2]) const
	{
		com[0] = centerOfMass[0];
		com[1] = centerOfMass[1];
	}

	void getLabPosition(Real com[2]) const
	{
		com[0] = labCenterOfMass[0];
		com[1] = labCenterOfMass[1];
	}

	Real getOrientation() const
	{
		return orientation;
	}

	virtual inline Real getMinRhoS() const
	{
		return rhoS;
	}

	virtual void outputSettings(ostream &outStream) const
	{
		outStream << "centerX " << center[0] << endl;
		outStream << "centerY " << center[1] << endl;
		outStream << "centerMassX " << centerOfMass[0] << endl;
		outStream << "centerMassY " << centerOfMass[1] << endl;
		outStream << "orientation " << orientation << endl;
		outStream << "rhoS " << rhoS << endl;
	}

	void removeMoments(const vector<BlockInfo>& vInfo)
	{
		const Real h   = vInfo[0].h_gridpoint;
	  const Real dv  = std::pow(vInfo[0].h_gridpoint, 2);
	  const Real eps = std::numeric_limits<Real>::epsilon();

    Real _M = 0, _J = 0, um = 0, vm = 0, am = 0, cx = 0, cy = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:_M,_J,um,vm,am,cx,cy)
  	for(int i=0; i<vInfo.size(); i++)
		{
  		const auto pos = obstacleBlocks.find(vInfo[i].blockID);
  		if(pos == obstacleBlocks.end()) continue;

  		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
  		for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				const Real chi = pos->second->chi[iy][ix];
  			if (chi == 0) continue;
				Real p[2];
  			vInfo[i].pos(p, ix, iy);
  			p[0] -= centerOfMass[0];
  			p[1] -= centerOfMass[1];
				const Real u_ = pos->second->udef[iy][ix][0];
				const Real v_ = pos->second->udef[iy][ix][1];
  			const Real rhochi = chi*pos->second->rho[iy][ix];
  			_M += rhochi;
  			um += rhochi*u_;
  			vm += rhochi*v_;
  			cx += rhochi*p[0];
  			cy += rhochi*p[1];
				am += rhochi*(p[0]*v_ - p[1]*u_);
				_J += rhochi*(p[0]*p[0] + p[1]*p[1]);
  		}
  	}

    um /= _M;
    vm /= _M;
    am /= _J;
    _M *= dv;
    _J *= dv;
		cx /= _M;
		cy /= _M;
		printf("Correction of: lin mom [%f %f] ang mom [%f]. Error in CM=[%f %f]\n", um, vm, am, cx-centerOfMass[0], cy-centerOfMass[1]);

    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<vInfo.size(); i++)
		{
      BlockInfo info = vInfo[i];
      const auto pos = obstacleBlocks.find(info.blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          Real p[2];
          info.pos(p, ix, iy);
          p[0] -= centerOfMass[0];
          p[1] -= centerOfMass[1];
          pos->second->udef[iy][ix][0] -= um -am*p[1];
          pos->second->udef[iy][ix][1] -= vm +am*p[0];
      }
    }
	};

	void computeVelocities(Real*const uBody, Real*const vBody, Real*const omegaBody, const vector<BlockInfo>& vInfo)
	{
    const Real h  = vInfo[0].h_gridpoint;
    const Real dv = std::pow(vInfo[0].h_gridpoint,2);

		Real _M = 0, _J = 0, um = 0, vm = 0, am = 0; //linear momenta
    #pragma omp parallel for schedule(dynamic) reduction(+:_M,_J,um,vm,am)
    for(int i=0; i<vInfo.size(); i++) {
        const BlockInfo info = vInfo[i];
        FluidBlock & b = *(FluidBlock*)info.ptrBlock;
        const auto pos = obstacleBlocks.find(info.blockID);
        if(pos == obstacleBlocks.end()) continue;

        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            const Real chi = pos->second->chi[iy][ix];
		  			if (chi == 0) continue;
						Real p[2];
		  			info.pos(p, ix, iy);
		  			p[0] -= centerOfMass[0];
		  			p[1] -= centerOfMass[1];

						const double rhochi = b(ix,iy).rho * chi;
						_M += rhochi;
						um += rhochi * b(ix,iy).u;
						vm += rhochi * b(ix,iy).v;
						_J += rhochi * (p[0]*p[0]       + p[1]*p[1]);
						am += rhochi * (p[0]*b(ix,iy).v - p[1]*b(ix,iy).u);
        }
    }

		*uBody 			= um / (_M+2.2e-16);
		*vBody 			= vm / (_M+2.2e-16);
		*omegaBody	= am / (_J+2.2e-16);
		J = _M / dv;
		M = _J / dv;
	}

	void penalize(const Real u, const Real v, const Real omega, const Real dt, const Real lambda, const vector<BlockInfo>& vInfo)
	{
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<vInfo.size(); i++) {
      FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real chi = pos->second->chi[iy][ix];
  			if (chi == 0) continue;
				Real p[2];
  			vInfo[i].pos(p, ix, iy);
  			p[0] -= centerOfMass[0];
  			p[1] -= centerOfMass[1];
				const Real alpha = 1./(1. + dt * lambda * chi);
				const Real uTot = u -omega*p[1] +pos->second->udef[iy][ix][0];
				const Real vTot = v +omega*p[0] +pos->second->udef[iy][ix][1];
				b(ix,iy).u = alpha*b(ix,iy).u + (1-alpha)*uTot;
	      b(ix,iy).v = alpha*b(ix,iy).v + (1-alpha)*vTot;
      }
    }
	}

	void _diagnostics(const Real uBody, const Real vBody, const Real omegaBody, const vector<BlockInfo>&vInfo, const Real nu, const Real time, const int step, const Real lambda)
	{
		double cX=0, cY=0, cmX=0, cmY=0, fx=0, fy=0, pMin=10, pMax=0, mass=0, volS=0, volF=0;
		const double dh = vInfo[0].h_gridpoint;

		#pragma omp parallel for reduction(max : pMax) reduction (min : pMin) reduction(+ : cX,cY,volF,cmX,cmY,fx,fy,mass,volS)
		for(int i=0; i<vInfo.size(); i++) {
			FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
			const auto pos = obstacleBlocks.find(vInfo[i].blockID);

			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
				double p[2] = {0,0};
				vInfo[i].pos(p, ix, iy);
				const Real chi = pos==obstacleBlocks.end()?0 : pos->second->chi[iy][ix];
				const Real rhochi = b(ix,iy).rho * chi;
				cmX += p[0] * rhochi; cX += p[0] * chi; fx += (b(ix,iy).u-uBody)*chi;
				cmY += p[1] * rhochi; cY += p[1] * chi; fy += (b(ix,iy).v-vBody)*chi;
				mass += rhochi; volS += chi; volF += (1-chi);
				pMin = min(pMin,(double)b(ix,iy).p);
				pMax = max(pMax,(double)b(ix,iy).p);
			}
		}

		cmX /= mass; cX /= volS; volS *= dh*dh; fx *= dh*dh*lambda;
		cmY /= mass; cY /= volS; volF *= dh*dh; fy *= dh*dh*lambda;
		const Real rhoSAvg = mass/volS, length = getCharLength();
		const Real speed = sqrt ( uBody * uBody + vBody * vBody);
		const Real cDx = 2*fx/(speed*speed*length + 2.2e-16);
		const Real cDy = 2*fy/(speed*speed*length + 2.2e-16);
		const Real Re_uBody = getCharLength()*speed/nu;
		const Real theta = getOrientation();
		Real CO[2], CM[2], labpos[2];
		getLabPosition(labpos);
		getCentroid(CO);
		getCenterOfMass(CM);

		stringstream ss;
		ss << "./_diagnostics.dat";
		ofstream myfile(ss.str(), fstream::app);
		if (!step)
		myfile<<"step time CO[0] CO[1] CM[0] CM[1] centroidX centroidY centerMassX centerMassY labpos[0] labpos[1] theta uBody[0] uBody[1] omegaBody Re_uBody cDx cDy rhoSAvg"<<endl;

		cout<<step<<" "<<time<<" "<<CO[0]<<" "<<CO[1]<<" "<<CM[0]<<" "<<CM[1]
			<<" " <<cX<<" "<<cY<<" "<<cmX<<" "<<cmY<<" "<<labpos[0]<<" "<<labpos[1]
			<<" "<<theta<<" "<<uBody<<" "<<vBody<<" "<<omegaBody<<" "<<Re_uBody
			<<" "<<cDx<<" "<<cDy<<" "<<rhoSAvg<<" "<<fx<<" "<<fy<<endl;

		myfile<<step<<" "<<time<<" "<<CO[0]<<" "<<CO[1]<<" "<<CM[0]<<" "<<CM[1]
			<<" " <<cX<<" "<<cY<<" "<<cmX<<" "<<cmY<<" "<<labpos[0]<<" "<<labpos[1]
			<<" "<<theta<<" "<<uBody<<" "<<vBody<<" "<<omegaBody<<" "<<Re_uBody
			<<" "<<cDx<<" "<<cDy<<" "<<rhoSAvg<<" "<<fx<<" "<<fy<<endl;

		myfile.close();
	}
};

class Disk : public Shape
{
 protected:
	Real radius;

 public:
	Disk(Real center[2], Real radius, const Real rhoS) :
	Shape(center, 0, rhoS), radius(radius) { }

	Real getCharLength() const override
	{
		return 2 * radius;
	}

	void outputSettings(ostream &outStream) const
	{
		outStream << "Disk\n";
		outStream << "radius " << radius << endl;

		Shape::outputSettings(outStream);
	}

	void create(const vector<BlockInfo>& vInfo) override
	{
		const Real h =  vInfo[0].h_gridpoint;
		for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

		FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    for(int i=0; i<vInfo.size(); i++) {
    	BlockInfo info = vInfo[i];
      //const auto pos = obstacleBlocks.find(info.blockID);
  		if(kernel._is_touching(info)) { //position of sphere + radius + 2*h safety
  			assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
  			obstacleBlocks[info.blockID] = new ObstacleBlock;
  			obstacleBlocks[info.blockID]->clear(); //memset 0
  		}
    }

		#pragma omp parallel
		{
			FillBlocks_Cylinder kernel(radius, h, center, rhoS);

	    #pragma omp for
			for(int i=0; i<vInfo.size(); i++) {
				BlockInfo info = vInfo[i];
				const auto pos = obstacleBlocks.find(info.blockID);
				if(pos == obstacleBlocks.end()) continue;
				FluidBlock& b = *(FluidBlock*)info.ptrBlock;
				kernel(info, b, pos->second);
			}
		}
	}
};

class DiskVarDensity : public Shape
{
 protected:
	const Real radius;
	const Real rhoTop;
	const Real rhoBot;

 public:
	DiskVarDensity(Real C[2], Real R, Real ang, Real rhoT, Real rhoB) :
	Shape(C, ang, min(rhoT,rhoB)), radius(R), rhoTop(rhoT), rhoBot(rhoB)
	{
		d_gm[0] = 0;
		// based on weighted average between the centers of mass of half-disks:
		d_gm[1] = -4.*radius/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

		centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
		centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
	}

	Real getCharLength() const  override
	{
		return 2 * radius;
	}

	void create(const vector<BlockInfo>& vInfo) override
	{
		const Real h =  vInfo[0].h_gridpoint;
		for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

		FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    for(int i=0; i<vInfo.size(); i++) {
    	BlockInfo info = vInfo[i];
      //const auto pos = obstacleBlocks.find(info.blockID);
  		if(kernel._is_touching(info)) { //position of sphere + radius + 2*h safety
  			assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
  			obstacleBlocks[info.blockID] = new ObstacleBlock;
  			obstacleBlocks[info.blockID]->clear(); //memset 0
  		}
    }

		#pragma omp parallel
		{
			FillBlocks_Cylinder kernelC(radius, h, center, rhoS);
			//assumption: if touches cylinder, it touches half cylinder:
			FillBlocks_HalfCylinder kernelH(radius, h, center, rhoS, orientation);

	    #pragma omp for
			for(int i=0; i<vInfo.size(); i++) {
				BlockInfo info = vInfo[i];
				const auto pos = obstacleBlocks.find(info.blockID);
				if(pos == obstacleBlocks.end()) continue;
				FluidBlock& b = *(FluidBlock*)info.ptrBlock;
				kernelC(info, b, pos->second);
				kernelH(info, b, pos->second);
			}
		}
	}

	void outputSettings(ostream &outStream)
	{
		outStream << "DiskVarDensity\n";
		outStream << "radius " << radius << endl;
		outStream << "rhoTop " << rhoTop << endl;
		outStream << "rhoBot " << rhoBot << endl;

		Shape::outputSettings(outStream);
	}
};

class Ellipse : public Shape
{
 protected:
	Real semiAxis[2] = {0,0};
	Real Torque = 0, old_Torque = 0, old_Dist = 100;
	Real powerOutput = 0, old_powerOutput = 0;

 public:
	Ellipse(Real C[2], Real SA[2], Real ang, const Real rho) :
    Shape(C, ang, rho)
  {
		semiAxis[0] = SA[0];
		semiAxis[1] = SA[1];
		printf("Created ellipse %f %f %f\n", semiAxis[0], semiAxis[1],rhoS); fflush(0);
  }

	Real getCharLength() const  override
	{
		return 2 * max(semiAxis[1],semiAxis[0]);
	}

	void outputSettings(ostream &outStream) const
	{
		outStream << "Ellipse\n";
		outStream << "semiAxisX " << semiAxis[0] << endl;
		outStream << "semiAxisY " << semiAxis[1] << endl;

		Shape::outputSettings(outStream);
	}

	#ifdef RL_MPI_CLIENT
	void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) override
	{
		assert(time_ptr not_eq nullptr);
		assert(communicator not_eq nullptr);
		if(!initialized_time_next_comm || *time_ptr>time_next_comm)
		{
			const Real w = *omegaBody, u = *uBody, v = *vBody;
			const Real cosAng = cos(orientation), sinAng = sin(orientation);
			const Real a=max(semiAxis[0],semiAxis[1]), b=min(semiAxis[0],semiAxis[1]);
			//Characteristic scales:
			const Real lengthscale = a, timescale = a/velscale;
			const Real velscale = std::sqrt((rhoS/1.-1)*9.8*b);
			//const Real torquescale = M_PI/8*pow((a*a-b*b)*velscale,2)*a/b;
			const Real torquescale = M_PI*lengthscale*lengthscale*velscale*velscale;
			//Nondimensionalization:
			const Real xdot = u/velscale, ydot = v/velscale;
			const Real X = labCenterOfMass[0]/a, Y = labCenterOfMass[1]/a;
			const Real U = xdot*cosAng +ydot*sinAng;
			const Real V = ydot*cosAng -xdot*sinAng;
			const Real W = w*timescale;
			const Real T = Torque/torquescale;
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

			vector<double> state = {U, V, W, X, Y, cosAng, sinAng, T, xdot, ydot}; vector<double> action = {0.};

	    printf("Sending (%lu) [%f %f %f %f %f %f %f %f %f %f], %f %f\n",
			state.size(),U,V,W,X,Y,cosAng,sinAng,T,xdot,ydot, Torque,torquescale);

			communicator->sendState(0, info, state, reward);

			if(info == _AGENT_LASTCOMM) abort();
			old_Dist = horzDist;
			old_Torque = Torque;
			old_powerOutput = powerOutput;
			initialized_time_next_comm = true;
			time_next_comm = time_next_comm + 0.5*timescale;
			info = _AGENT_NORMALCOMM;

			communicator->recvAction(action);
	       printf("Received %f\n", action[0]);
			Torque = action[0]*torquescale;
		}

		*omegaBody += dt*Torque/J;
		powerOutput += dt*Torque*Torque;
	}
	#endif

	void create(const vector<BlockInfo>& vInfo) override
	{
		const Real h =  vInfo[0].h_gridpoint;
		for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

		FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);
    for(int i=0; i<vInfo.size(); i++) {
    	BlockInfo info = vInfo[i];
      //const auto pos = obstacleBlocks.find(info.blockID);
  		if(kernel._is_touching(info)) { //position of sphere + radius + 2*h safety
  			assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
  			obstacleBlocks[info.blockID] = new ObstacleBlock;
  			obstacleBlocks[info.blockID]->clear(); //memset 0
  		}
    }

		#pragma omp parallel
		{
			FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);

	    #pragma omp for
			for(int i=0; i<vInfo.size(); i++) {
				BlockInfo info = vInfo[i];
				const auto pos = obstacleBlocks.find(info.blockID);
				if(pos == obstacleBlocks.end()) continue;
				FluidBlock& b = *(FluidBlock*)info.ptrBlock;
				kernel(info, b, pos->second);
			}
		}
	}
};

/*
class EllipseVarDensity : public Shape
{
 protected:
	// these quantities are defined in the local coordinates of the ellipse
	Real semiAxis[2];
	Real rhoS1, rhoS2;

	// code from http://www.geometrictools.com/
	//----------------------------------------------------------------------------
	// The ellipse is (x0/semiAxis0)^2 + (x1/semiAxis1)^2 = 1.  The query point is (y0,y1).
	// The function returns the distance from the query point to the ellipse.
	// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
	//----------------------------------------------------------------------------
	Real DistancePointEllipseSpecial (const Real e[2], const Real y[2], Real x[2]) const
	{
		Real distance = (Real)0;
		if (y[1] > (Real)0)
		{
			if (y[0] > (Real)0)
			{
				// Bisect to compute the root of F(t) for t >= -e1*e1.
				Real esqr[2] = { e[0]*e[0], e[1]*e[1] };
				Real ey[2] = { e[0]*y[0], e[1]*y[1] };
				Real t0 = -esqr[1] + ey[1];
				Real t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
				Real t = t0;
				const int imax = 2*std::numeric_limits<Real>::max_exponent;
				for (int i = 0; i < imax; ++i)
				{
					t = ((Real)0.5)*(t0 + t1);
					if (t == t0 || t == t1)
					{
						break;
					}

					Real r[2] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]) };
					Real f = r[0]*r[0] + r[1]*r[1] - (Real)1;
					if (f > (Real)0)
					{
						t0 = t;
					}
					else if (f < (Real)0)
					{
						t1 = t;
					}
					else
					{
						break;
					}
				}

				x[0] = esqr[0]*y[0]/(t + esqr[0]);
				x[1] = esqr[1]*y[1]/(t + esqr[1]);
				Real d[2] = { x[0] - y[0], x[1] - y[1] };
				distance = sqrt(d[0]*d[0] + d[1]*d[1]);
			}
			else  // y0 == 0
			{
				x[0] = (Real)0;
				x[1] = e[1];
				distance = fabs(y[1] - e[1]);
			}
		}
		else  // y1 == 0
		{
			Real denom0 = e[0]*e[0] - e[1]*e[1];
			Real e0y0 = e[0]*y[0];
			if (e0y0 < denom0)
			{
				// y0 is inside the subinterval.
				Real x0de0 = e0y0/denom0;
				Real x0de0sqr = x0de0*x0de0;
				x[0] = e[0]*x0de0;
				x[1] = e[1]*sqrt(fabs((Real)1 - x0de0sqr));
				Real d0 = x[0] - y[0];
				distance = sqrt(d0*d0 + x[1]*x[1]);
			}
			else
			{
				// y0 is outside the subinterval.  The closest ellipse point has
				// x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
				x[0] = e[0];
				x[1] = (Real)0;
				distance = fabs(y[0] - e[0]);
			}
		}
		return distance;
	}

	Real DistancePointEllipse(const Real y[2], Real x[2]) const
	{
		// Determine reflections for y to the first quadrant.
		bool reflect[2];
		int i, j;
		for (i = 0; i < 2; ++i)
		{
			reflect[i] = (y[i] < (Real)0);
		}

		// Determine the axis order for decreasing extents.
		int permute[2];
		if (semiAxis[0] < semiAxis[1])
		{
			permute[0] = 1;  permute[1] = 0;
		}
		else
		{
			permute[0] = 0;  permute[1] = 1;
		}

		int invpermute[2];
		for (i = 0; i < 2; ++i)
		{
			invpermute[permute[i]] = i;
		}

		Real locE[2], locY[2];
		for (i = 0; i < 2; ++i)
		{
			j = permute[i];
			locE[i] = semiAxis[j];
			locY[i] = y[j];
			if (reflect[j])
			{
				locY[i] = -locY[i];
			}
		}

		Real locX[2];
		Real distance = DistancePointEllipseSpecial(locE, locY, locX);

		// Restore the axis order and reflections.
		for (i = 0; i < 2; ++i)
		{
			j = invpermute[i];
			if (reflect[j])
			{
				locX[j] = -locX[j];
			}
			x[i] = locX[j];
		}

		return distance;
	}

 public:
	EllipseVarDensity(Real center[2], Real semiAxis[2], Real orientation, const Real rhoS1, const Real rhoS2, const Real mollChi, const Real mollRho, bool bPeriodic[2], Real domainSize[2]) : Shape(center, orientation, min(rhoS1,rhoS2), mollChi, mollRho, bPeriodic, domainSize), semiAxis{semiAxis[0],semiAxis[1]}, rhoS1(rhoS1), rhoS2(rhoS2)
	{
		d_gm[0] = 0;
		d_gm[1] = -4.*semiAxis[0]/(3.*M_PI) * (rhoS1-rhoS2)/(rhoS1+rhoS2); // based on weighted average between the centers of mass of half-disks

		centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
		centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
	}

	Real chi(Real p[2], Real h) const
	{
		const Real centerPeriodic[2] = {center[0] - floor(center[0]/domainSize[0]) * bPeriodic[0],
										center[1] - floor(center[1]/domainSize[1]) * bPeriodic[1]};
		Real x[2] = {0,0};
		const Real pShift[2] = {p[0]-centerPeriodic[0],p[1]-centerPeriodic[1]};

		const Real rotatedP[2] = { cos(orientation)*pShift[1] - sin(orientation)*pShift[0],
								   sin(orientation)*pShift[1] + cos(orientation)*pShift[0] };
		const Real dist = DistancePointEllipse(rotatedP, x);
		const int sign = ( (rotatedP[0]*rotatedP[0]+rotatedP[1]*rotatedP[1]) > (x[0]*x[0]+x[1]*x[1]) ) ? 1 : -1;

		return smoothHeaviside(sign*dist,0,mollChi*sqrt(2)*h);
	}

	Real rho(Real p[2], Real h, Real mask) const
	{
		// not handling periodicity

		Real r = 0;
		if (orientation == 0 || orientation == 2*M_PI)
			r = smoothHeaviside(p[1],center[1], mollRho*sqrt(2)*h);
		else if (orientation == M_PI)
			r = smoothHeaviside(center[1],p[1], mollRho*sqrt(2)*h);
		else if (orientation == M_PI_2)
			r = smoothHeaviside(center[0],p[0], mollRho*sqrt(2)*h);
		else if (orientation == 3*M_PI_2)
			r = smoothHeaviside(p[0],center[0], mollRho*sqrt(2)*h);
		else
		{
			const Real tantheta = tan(orientation);
			r = smoothHeaviside(p[1], tantheta*p[0]+center[1]-tantheta*center[0], mollRho*sqrt(2)*h);
			r = (orientation>M_PI_2 && orientation<3*M_PI_2) ? 1-r : r;
		}

		return ((rhoS2-rhoS1)*r+rhoS1)*mask + 1.*(1.-mask);
	}

	Real rho(Real p[2], Real h) const
	{
		Real mask = chi(p,h);
		return rho(p,h,mask);
	}

	Real getCharLength() const
	{
		return 2 * semiAxis[1];
	}

	void outputSettings(ostream &outStream) const
	{
		outStream << "Ellipse\n";
		outStream << "semiAxisX " << semiAxis[0] << endl;
		outStream << "semiAxisY " << semiAxis[1] << endl;
		outStream << "rhoS1 " << rhoS1 << endl;
		outStream << "rhoS2 " << rhoS2 << endl;

		Shape::outputSettings(outStream);
	}
};
*/
#endif

//
//  Simulation_FSI.h
//  CubismUP_2D
//
//	Base class for Fluid-Structure Interaction (FSI) simulations from which any FSI simulation case should inherit
//	Contains the base structure and interface that any FSI simulation class should have
//	Inherits from Simulation_Fluid
//	Assumes use of Penalization to handle rigid body
//
//  Created by Christian Conti on 3/25/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_Simulation_FSI_h
#define CubismUP_2D_Simulation_FSI_h

#include "Simulation_Fluid.h"
#include "ShapesSimple.h"
#include "Blowfish.h"
#include <random>

class Simulation_FSI : public Simulation_Fluid
{
 protected:
	// penalization parameter
	Real lambda, dlm;
	Real dragP[2], dragV;

	// body
	Shape* shape;

 public:
	Simulation_FSI(const int argc, const char ** argv) : Simulation_Fluid(argc,argv) { }

	virtual void init()
	{
		Simulation_Fluid::init();

		lambda = parser("-lambda").asDouble(1e5);
		dlm = parser("-dlm").asDouble(1.);
		double rhoS = parser("-rhoS").asDouble(1);
		Real centerOfMass[2] = {0,0};
    vector<BlockInfo> vInfo = grid->getBlocksInfo();

		string shapeType = parser("-shape").asString("disk");
		if (shapeType=="disk")
		{
			Real radius = parser("-radius").asDouble(0.1);
			shape = new Disk(centerOfMass, radius, rhoS);
		}
		else if (shapeType=="ellipse")
		{
			Real semiAxis[2] = {parser("-semiAxisX").asDouble(.1), parser("-semiAxisY").asDouble(.2)};
			Real angle = parser("-angle").asDouble(0.0);
			shape = new Ellipse(centerOfMass, semiAxis, angle, rhoS);
		}
		else if (shapeType=="diskVarDensity")
		{
			Real radius = parser("-radius").asDouble(0.1);
			Real rhoTop = parser("-rhoTop").asDouble(1);
			Real angle = parser("-angle").asDouble(0.0);
			shape = new DiskVarDensity(centerOfMass, radius, angle, rhoTop, rhoS);
		}
		else if (shapeType=="blowfish")
		{
			#ifndef RL_MPI_CLIENT
				Real angle = parser("-angle").asDouble(0.0);
			#else
			  mt19937 gen(parser("-Socket").asInt(0));
				uniform_real_distribution<Real> dis(-.1,.1);
				Real angle = dis(gen);
			#endif
			Real radius = parser("-radius").asDouble(0.1);
			shape = new Blowfish(centerOfMass, angle, radius);
		}
		else
		{
			cout << "Error - this shape is not currently implemented! Aborting now\n";
			abort();
		}
		// nothing needs to be done on restart
	}

	virtual ~Simulation_FSI()
	{
		delete shape;
	}
};

#endif

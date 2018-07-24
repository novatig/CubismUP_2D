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
 public:

  Shape* getShape() { return sim.shapes[0]; }

	Simulation_FSI(const int argc, char ** argv) : Simulation_Fluid(argc,argv) { }

	virtual void init()
	{
		Simulation_Fluid::init();

		parser.set_strict_mode();
    const Real axX = parser("-bpdx").asInt();
    const Real axY = parser("-bpdy").asInt();
		parser.unset_strict_mode();
    const Real ext = std::max(axX, axY);
    Real center[2] = {
        (Real) parser("-xpos").asDouble(.5*axX/ext),
        (Real) parser("-ypos").asDouble(.5*axY/ext)
    };

    Shape* shape = nullptr;
		const string shapeType = parser("-shape").asString("disk");
		if (shapeType=="disk")
      shape = new Disk(sim, parser, center);
		else if (shapeType=="ellipse")
      shape = new Ellipse(sim, parser, center);
		else if (shapeType=="diskVarDensity")
      shape = new DiskVarDensity(sim, parser, center);
		else if (shapeType=="blowfish")
			shape = new Blowfish(sim, parser, center);
		else
		{
			cout << "Error - this shape is not currently implemented! Aborting now\n";
			abort();
		}
    assert(shape not_eq nullptr);
    sim.shapes.push_back(shape);

		// nothing needs to be done on restart
	}
};

#endif

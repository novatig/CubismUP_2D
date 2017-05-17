//
//  Sim_FSI_Moving.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef __CubismUP_2D__Sim_FSI_Moving__
#define __CubismUP_2D__Sim_FSI_Moving__

#include "Simulation_FSI.h"

class Sim_FSI_Moving : public Simulation_FSI
{
protected:
	Real uBody[2], omegaBody;
	double dtBody, dtCFL, dtLCFL, dtFourier;
	double re, nu;
	
	void _diagnostics();
	void _ic();
	double _nonDimensionalTime();
	
	void _outputSettings(ostream& outStream);
	void _inputSettings(istream& inStream);
	
public:
	Sim_FSI_Moving(const int argc, const char ** argv);
	virtual ~Sim_FSI_Moving();
	
	void init();
	void simulate();
};

#endif /* defined(__CubismUP_2D__Sim_FSI_Moving__) */

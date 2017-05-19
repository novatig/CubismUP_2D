//
//  Sim_FSI_Oscillating.h
//  CubismUP_2D
//
//  Created by Christian Conti on 11/23/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef __CubismUP_2D__Sim_FSI_Oscillating__
#define __CubismUP_2D__Sim_FSI_Oscillating__

#include "Simulation_FSI.h"

class Sim_FSI_Oscillating : public Simulation_FSI
{
protected:
	Real uBody[2], omegaBody;
	double dtBody, dtCFL, dtLCFL, dtFourier;
	double re, kc, nu, umax, freq; // Keulegan-Carpenter number
	
	void _diagnostics();
	void _ic();
	double _nonDimensionalTime();
	
	void _outputSettings(ostream& outStream);
	void _inputSettings(istream& inStream);
	
public:
	Sim_FSI_Oscillating(const int argc, const char ** argv);
	virtual ~Sim_FSI_Oscillating();
	
	void init();
	void simulate();
};

#endif /* defined(__CubismUP_2D__Sim_FSI_Oscillating__) */

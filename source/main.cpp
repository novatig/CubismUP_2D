//
//  main.cpp
//  CubismUP_2D
//
//  Created by Christian Conti on 1/7/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
using namespace std;

#include "Definitions.h"

#include "Simulation_Fluid.h"
#include "Sim_FSI_Gravity.h"
//#include "Sim_RayleighTaylor.h"
//#include "Sim_Bubble.h"
//#include "Sim_Jet.h"

int main(int argc, const char **argv)
{
	{
		cout << "====================================================================================================================\n";
		cout << "\t\tCubism UP 2D (velocity-pressure 2D incompressible Navier-Stokes solver)\n";
		cout << "====================================================================================================================\n";
	}

	ArgumentParser parser(argc,argv);
	parser.set_strict_mode();

	#ifdef RL_MPI_CLIENT
		const int socket  = parser("-Socket").asInt(-1);
		const int nStates = parser("-nStates").asInt(-1);
		const int nAction = parser("-nAction").asInt(-1);
      const bool bRL = socket>0 && nStates>0 && nAction>0;
		Communicator* const communicator = bRL ? new Communicator(socket,nStates,nAction) : nullptr;
      if(bRL){
		   printf("Starting communication with RL over socket %d\n",socket); fflush(0);
      }

		Sim_FSI_Gravity* sim = new Sim_FSI_Gravity(communicator, argc, argv);
	#else
		Sim_FSI_Gravity* sim = new Sim_FSI_Gravity(argc, argv);
	#endif

	sim->init();
	sim->simulate();

	return 0;
}

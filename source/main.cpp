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
#include "Sim_FSI_Fixed.h"
#include "Sim_FSI_Moving.h"
#include "Sim_FSI_Oscillating.h"
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
	
	//string simSetting = parser("-sim").asString();
	//Simulation_Fluid * sim;
	/*
	if (simSetting=="fixed")
		sim = new Sim_FSI_Fixed(argc, argv);
	else if (simSetting=="moving")
		sim = new Sim_FSI_Moving(argc, argv);
	else if (simSetting=="oscillating")
		sim = new Sim_FSI_Oscillating(argc, argv);
	else if (simSetting=="falling")
	*/
	Sim_FSI_Gravity* sim = new Sim_FSI_Gravity(argc, argv);
	/*
	else if (simSetting=="rti")
		sim = new Sim_RayleighTaylor(argc, argv);
	else if (simSetting=="bubble")
		sim = new Sim_Bubble(argc, argv);
	else if (simSetting=="jet")
		sim = new Sim_Jet(argc, argv);
	else
		throw std::invalid_argument("This simulation setting does not exist!");
	*/
	sim->init();
	sim->simulate();
	
	return 0;
}

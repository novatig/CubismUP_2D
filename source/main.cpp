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

int main(int argc, char **argv)
{
  cout<<"=======================================================================================\n";
  cout<<"\tCubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)\n";
  cout<<"=======================================================================================\n";

  ArgumentParser parser(argc,argv);
  parser.set_strict_mode();

  Sim_FSI_Gravity* sim = new Sim_FSI_Gravity(argc, argv);
  sim->init();
  sim->simulate();

  return 0;
}

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

#include "Simulation.h"

int main(int argc, char **argv)
{
  for(int i=0; i<argc; i++) {printf("%s\n",argv[i]); fflush(0);}
  cout<<"===================================================================\n";
  cout<<"  CubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)  \n";
  cout<<"===================================================================\n";

  ArgumentParser parser(argc,argv);
  parser.set_strict_mode();

  Simulation* sim = new Simulation(argc, argv);
  sim->init();
  sim->simulate();

  return 0;
}

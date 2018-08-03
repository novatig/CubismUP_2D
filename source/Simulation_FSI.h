//
//  Simulation_FSI.h
//  CubismUP_2D
//
//  Base class for Fluid-Structure Interaction (FSI) simulations from which any FSI simulation case should inherit
//  Contains the base structure and interface that any FSI simulation class should have
//  Inherits from Simulation_Fluid
//  Assumes use of Penalization to handle rigid body
//
//  Created by Christian Conti on 3/25/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "Simulation_Fluid.h"
#include <random>

class Simulation_FSI : public Simulation_Fluid
{
 public:



  Simulation_FSI(const int argc, char ** argv) : Simulation_Fluid(argc,argv) { }

  virtual void init()
  {
    Simulation_Fluid::init();


  }
};

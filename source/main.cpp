//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Simulation.h"

#include "mpi.h"

int main(int argc, char **argv)
{
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  for(int i=0; i<argc; i++) {printf("%s\n",argv[i]); fflush(0);}
  std::cout
  <<"=======================================================================\n";
  std::cout
  <<"    CubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)    \n";
  std::cout
  <<"=======================================================================\n";

  ArgumentParser parser(argc,argv);
  parser.set_strict_mode();

  Simulation* sim = new Simulation(argc, argv);
  sim->init();
  sim->simulate();

  MPI_Finalize();
  return 0;
}

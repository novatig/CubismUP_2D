//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
//#include <random>

#include "Communicator.h"
#include "Simulation.h"
#include "Obstacles/Glider.h"

#include "mpi.h"
using namespace cubism;

inline void resetIC(Glider* const agent, Communicator*const c) {
  const Real A = 5*M_PI/180; // start between -5 and 5 degrees
  std::uniform_real_distribution<Real> dis(-A, A);
  const auto SA = c->isTraining() ? dis(c->getPRNG()) : 0;
  agent->setOrientation(SA);
}

int app_main(
  Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv,    // arguments read from app's runtime settings file
  const unsigned numSteps  // number of time steps to run before exit
)
{
  printf("Simulating for %u steps accoding to settings:\n", numSteps);
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
  const int nActions = 1, nStates = 10;
  comm->update_state_action_dims(nStates, nActions);
  std::vector<double> upper_action_bound{1}, lower_action_bound{-1};
  comm->set_action_scales(upper_action_bound, lower_action_bound, true);
  std::vector<bool> b_observable = {1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
  comm->set_state_observable(b_observable);
  const unsigned maxLearnStepPerSim = numSteps;

  Simulation sim(argc, argv);
  sim.init();

  Glider* const agent = dynamic_cast<Glider*>( sim.getShapes()[0] );
  if(agent==nullptr) { printf("Obstacle was not a Glider!\n"); abort(); }
  if(comm->isTraining() == false) {
    sim.sim.verbose = true; sim.sim.muteAll = false;
    sim.sim.dumpTime = agent->timescale / 20;
  }
  unsigned sim_id = 0, tot_steps = 0;

  // Terminate loop if reached max number of time steps. Never terminate if 0
  while( numSteps == 0 || tot_steps<numSteps ) // train loop
  {
    if(comm->isTraining() == false)
    {
      char dirname[1024]; dirname[1023] = '\0';
      sprintf(dirname, "run_%08u/", sim_id);
      printf("Starting a new sim in directory %s\n", dirname);
      mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      chdir(dirname);
    }

    sim.reset();
    resetIC(agent, comm); // randomize initial conditions

    double t = 0;
    unsigned step = 0;
    bool agentOver = false;

    comm->sendInitState( agent->state() ); //send initial state

    while (true) //simulation loop
    {
      agent->act(comm->recvAction());
      const double tNextAct = (step+1) * agent->DTactions;
      while (t < tNextAct)
      {
        const double dt = sim.calcMaxTimestep();
        t += dt;

        if ( sim.advance( dt ) ) { // if true sim has ended
          printf("Set -tend 0. This file decides the length of train sim.\n");
          assert(false); fflush(0); abort();
        }
        if ( agent->isOver() ) {
          agentOver = true;
          break;
        }
      }
      step++;
      tot_steps++;
      const std::vector<double> state = agent->state();
      const double reward= agentOver? agent->terminalReward() : agent->reward();

      if (agentOver) {
        printf("Agent failed\n"); fflush(0);
        comm->sendTermState(state, reward);
        break;
      }
      else
      if (step >= maxLearnStepPerSim) {
        printf("Sim ended, should not happen for this env\n"); fflush(0);
        comm->truncateSeq(state, reward);
        break;
      }
      else comm->sendState(state, reward);
    } // simulation is done

    if(comm->isTraining() == false) chdir("../");
    sim_id++;
  }

  return 0;
}

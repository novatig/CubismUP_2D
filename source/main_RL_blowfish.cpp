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

#include "Communicator.h"
#include "Simulation.h"
#include "BlowFish.h"

#include "mpi.h"
//
// All these functions are defined here and not in object itself because
// Many different tasks, each requiring different state/act/rew descriptors
// could be designed for the same objects (also to fully encapsulate RL).
//
// main hyperparameters:
// number of actions per characteristic time scale
// max number of actions per simulation
// range of angles in initial conditions

inline void resetIC(BlowFish* const agent, std::mt19937& gen) {
  const Real A = 10*M_PI/180; // start between -15 and 15 degrees
  uniform_real_distribution<Real> dis(-A, A);
  agent->setOrientation(dis(gen));
}
inline void setAction(BlowFish* const agent, const vector<double> act) {
  agent->flapAcc_R = act[0]/agent->timescale/agent->timescale;
  agent->flapAcc_L = act[1]/agent->timescale/agent->timescale;
}
inline vector<double> getState(const BlowFish* const agent) {
  const double velscale = agent->radius / agent->timescale;
  const double w = agent->getW() * agent->timescale;
  const double angle = agent->getOrientation();
  const double u = agent->getU() / velscale;
  const double v = agent->getV() / velscale;
  const double cosAng = std::cos(angle), sinAng = std::sin(angle);
  const double U = u*cosAng + v*sinAng, V = v*cosAng - u*sinAng;
  const double WR = agent->flapVel_R * agent->timescale;
  const double WL = agent->flapVel_L * agent->timescale;
  const double AR = agent->flapAng_R, AL = agent->flapAng_L;
  vector<double> states = {U, V, w, angle, AR, AL, WR, WL};
  printf("Sending [%f %f %f %f %f %f %f %f]\n", U,V,w,angle,AR,AL,WR,WL);
  return states;
}
inline double getReward(const BlowFish* const agent) {
  const double velscale = agent->radius / agent->timescale;
  const double angle = agent->getOrientation();
  const double u = agent->getU() / velscale;
  const double v = agent->getV() / velscale;
  const double cosAng = std::cos(angle);
  const bool ended = cosAng<0;
  const double reward = ended ? -10 : cosAng -std::sqrt(u*u+v*v);
  return reward;
}
inline bool isTerminal(const BlowFish* const agent) {
  const Real angle = agent->getOrientation();
  const Real cosAng = std::cos(angle);
  return cosAng<0;
}
inline double getTimeToNextAct(const BlowFish* const agent, const double t) {
  return t + agent->timescale / 4;
}

int app_main(
  Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv,    // arguments read from app's runtime settings file
  const unsigned numSteps  // number of time steps to run before exit
) {
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
  const int nActions = 2, nStates = 8;
  const unsigned maxLearnStepPerSim = 500; // random number... TODO
  Communicator& communicator = *comm;
  communicator.update_state_action_dims(nStates, nActions);

  Simulation sim(argc, argv);
  sim.init();

  BlowFish* const agent = dynamic_cast<BlowFish*>( sim.getShapes()[0] );
  if(agent==nullptr) { printf("Obstacle was not a BlowFish!\n"); abort(); }
  //sim.sim.dumpTime = agent->timescale / 10; // to force dumping
  char dirname[1024]; dirname[1023] = '\0';
  unsigned sim_id = 0, tot_steps = 0;

  // Terminate loop if reached max number of time steps. Never terminate if 0
  while( numSteps == 0 || tot_steps<numSteps ) // train loop
  {
    sprintf(dirname, "run_%08u/", sim_id);
    printf("Starting a new sim in directory %s\n", dirname);
    mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    chdir(dirname);
    sim.reset();
    resetIC(agent, communicator.gen); // randomize initial conditions

    communicator.sendInitState(getState(agent)); //send initial state

    Real t = 0;
    unsigned step = 0;
    bool agentOver = false;

    while (true) //simulation loop
    {
      setAction(agent, communicator.recvAction());
      const double tNextAct = getTimeToNextAct(agent, t);
      while (t < tNextAct)
      {
        const double dt = sim.calcMaxTimestep();
        t += dt;

        if ( sim.advance( dt ) ) { // if true sim has ended
          printf("Set -tend 0. This file decides the length of train sim.\n");
          assert(false); fflush(0); abort();
        }
        if ( isTerminal(agent) ) {
          agentOver = true;
          break;
        }
      }
      step++;
      tot_steps++;
      const vector<double> state = getState(agent);
      const double reward = getReward(agent);

      if (agentOver) {
        printf("Agent failed\n"); fflush(0);
        communicator.sendTermState(state, reward);
        break;
      }
      else
      if (step >= maxLearnStepPerSim) {
        printf("Sim ended\n"); fflush(0);
        communicator.truncateSeq(state, reward);
        break;
      }
      else communicator.sendState(state, reward);
    } // simulation is done
    chdir("../");
    sim_id++;
  }

  return 0;
}

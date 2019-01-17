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
#include "SmartCylinder.h"

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
#define FREQ_ACTIONS 1

inline void resetIC(
  SmartCylinder*const a, Shape*const p, Communicator*const c)
{
  uniform_real_distribution<double> dis(-2.5, 2.5);
  //const double SX = c->isTraining()? dis(c->getPRNG()) : 0;
  //const double SY = c->isTraining()? dis(c->getPRNG()) : 0;
  const double SX = c->isTraining()? dis(c->getPRNG()) : 0;
  const double SY = c->isTraining()? dis(c->getPRNG()) : 0;
  const Real L = a->getCharLength()/2, OX = p->center[0], OY = p->center[1];
  double C[2] = { OX + (4+SX)*L, OY + SY*L };
  if(a->bFixedy) {
    const Real deltaY = C[1]-OY, MA = a->getCharMass(), MP = p->getCharMass();
    p->centerOfMass[1] = OY - deltaY * MA / (MA + MP);
    p->center[1] = OY - deltaY * MA / (MA + MP);
    C[1] = OY + deltaY * MP / (MA + MP);
  }
  #ifdef SMART_ELLIPSE
    const double SA = c->isTraining()? dis(c->getPRNG()) : 0;
    a->setOrientation(SA*M_PI/10);
  #endif
  a->setCenterOfMass(C);
}

inline void setAction(
  SmartCylinder*const agent, const Shape*const p,
  const std::vector<double> act, const double t)
{
  agent->act(act, p->getCharSpeed());
}

inline std::vector<double> getState(
  const SmartCylinder*const a, const Shape*const p, const double t)
{
  const auto S = a->state(p->center[0], p->center[1], p->getCharSpeed());
  printf("t:%f s:%f %f, %f %f %f, %f %f %f %f %f %f %f %f\n ",
    t, S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[8],S[9],S[10],S[11],S[12]);
  return S;
}

inline bool isTerminal(
  const SmartCylinder*const a, const Shape*const p)
{
  const Real L = a->getCharLength()/2, OX = p->center[0], OY = p->center[1];
  const double X = (a->center[0]-OX)/ L, Y = (a->center[1]-OY)/ L;
  return X<1 || X>7 || std::fabs(Y)>3;
}

inline double getReward(
  SmartCylinder*const a, const Shape*const p)
{
  const Real energy = a->reward(p->getCharSpeed()); // force call to reset
  return isTerminal(a, p)? -100 : energy;
}

inline double getTimeToNextAct(
  const SmartCylinder*const agent, const Shape*const p, const double t)
{
  return t + FREQ_ACTIONS * agent->getCharLength() / p->getCharSpeed();
}

inline bool checkNaN(std::vector<double>& state, double& reward)
{
  bool bTrouble = false;
  if(std::isnan(reward)) bTrouble = true;
  for(size_t i=0; i<state.size(); i++) if(std::isnan(state[i])) bTrouble = true;
  if ( bTrouble )
  {
    reward = -100;
    printf("Caught a nan!\n");
    state = std::vector<double>(state.size(), 0);
  }
  return bTrouble;
}

int app_main(
  Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv,    // arguments read from app's runtime settings file
  const unsigned numSteps  // number of time steps to run before exit
)
{
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
  #ifdef SMART_ELLIPSE
  const int nActions = 3, nStates = 7 + 8;
  #else
  const int nActions = 3, nStates = 5 + 8;
  #endif
  const unsigned maxLearnStepPerSim = 400; // random number... TODO

  comm->update_state_action_dims(nStates, nActions);
  // Tell smarties that action space should be bounded.
  // First action modifies curvature, only makes sense between -1 and 1
  // Second action affects Tp = (1+act[1])*Tperiod_0 (eg. halved if act[1]=-.5).
  // If too small Re=L^2*Tp/nu would increase too much, we allow it to
  //  double at most, therefore we set the bounds between -0.5 and 0.5.
  //vector<double> upper_action_bound{1.0, 0.25}, lower_action_bound{-1., -.25};
  //comm->set_action_scales(upper_action_bound, lower_action_bound, true);

  Simulation sim(argc, argv);
  sim.init();

  Shape * const object = sim.getShapes()[0];
  SmartCylinder*const agent = dynamic_cast<SmartCylinder*>(sim.getShapes()[1]);
  if(agent==nullptr) { printf("Agent was not a SmartCylinder!\n"); abort(); }

  if(comm->isTraining() == false) {
    //sim.sim.verbose = true;
    sim.sim.muteAll = false;
    sim.sim.dumpTime = 0.25;
  }
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
    resetIC(agent, object, comm); // randomize initial conditions
    sim.reinit();

    double t = 0, tNextAct = 0;
    unsigned step = 0;
    bool agentOver = false;

    comm->sendInitState(getState(agent,object,t)); //send initial state

    while (true) //simulation loop
    {
      setAction(agent, object, comm->recvAction(), tNextAct);
      tNextAct = getTimeToNextAct(agent,object,tNextAct);
      while (t < tNextAct)
      {
        const double dt = sim.calcMaxTimestep();
        t += dt;

        if ( sim.advance( dt ) ) { // if true sim has ended
          printf("Set -tend 0. This file decides the length of train sim.\n");
          assert(false); fflush(0); abort();
        }
        if ( isTerminal(agent,object) ) {
          agentOver = true;
          break;
        }
      }
      step++;
      tot_steps++;
      vector<double> state = getState(agent,object,t);
      double reward = getReward(agent,object);

      if ( agentOver || checkNaN(state, reward) ) {
        printf("Agent failed\n"); fflush(0);
        comm->sendTermState(state, reward);
        break;
      }
      else
      if (step >= maxLearnStepPerSim) {
        printf("Sim ended\n"); fflush(0);
        comm->truncateSeq(state, reward);
        break;
      }
      else comm->sendState(state, reward);
    } // simulation is done
    chdir("../");
    sim_id++;
  }

  return 0;
}

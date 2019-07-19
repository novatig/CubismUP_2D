//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include <unistd.h>
#include <sys/stat.h>

#include "Communicators/Communicator_MPI.h"
#include "Simulation.h"
#include "Obstacles/StefanFish.h"

using namespace cubism;
//
// All these functions are defined here and not in object itself because
// Many different tasks, each requiring different state/act/rew descriptors
// could be designed for the same objects (also to fully encapsulate RL).
//
// main hyperparameters:
// number of actions per characteristic time scale
// max number of actions per simulation
// range of angles in initial conditions

inline void resetIC(StefanFish* const a, Shape*const p,
                    smarties::Communicator*const c)
{
  std::uniform_real_distribution<double> disA(-20./180.*M_PI, 20./180.*M_PI);
  std::uniform_real_distribution<double> disX(0, 0.5),  disY(-0.25, 0.25);
  const double SX = c->isTraining()? disX(c->getPRNG()) : 0.25;
  const double SY = c->isTraining()? disY(c->getPRNG()) : 0.00;
  const double SA = c->isTraining()? disA(c->getPRNG()) : 0.00;
  double C[2] = { p->center[0] + (1+SX)*a->length,
                  p->center[1]     + SY*a->length };
  p->centerOfMass[1] = p->center[1] - ( C[1] - p->center[1] );
  p->center[1] = p->center[1] - ( C[1] - p->center[1] );
  a->setCenterOfMass(C);
  a->setOrientation(SA);
}

inline void setAction(StefanFish* const agent,
  const std::vector<double> act, const double t)
{
  agent->act(t, act);
}

inline bool isTerminal(const StefanFish*const a, const Shape*const p) {
  const double X = ( a->center[0] - p->center[0] )/ a->length;
  const double Y = ( a->center[1] - p->center[1] )/ a->length;
  assert(X>0);
  return std::fabs(Y)>1 || X<0.5 || X>3;
}

inline double getReward(const StefanFish* const a, const Shape*const p) {
  //double efficiency = a->reward(); version using member-function (PW)
  return isTerminal(a, p)? -10 : a->EffPDefBnd; //efficiency;
}

inline bool checkNaN(std::vector<double>& state, double& reward)
{
  bool bTrouble = false;
  if(std::isnan(reward)) bTrouble = true;
  for(size_t i=0; i<state.size(); i++) if(std::isnan(state[i])) bTrouble = true;
  if ( bTrouble )
  {
    reward = -1000;
    printf("Caught a nan!\n");
    state = std::vector<double>(state.size(), 0);
  }
  return bTrouble;
}

inline double getTimeToNextAct(const StefanFish* const agent, const double t) {
  return t + agent->getLearnTPeriod() / 2;
}

int app_main(
  smarties::Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,                  // mpi_comm that mpi-based apps can use
  int argc, char**argv             // args read from app's runtime settings file
) {
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
  #ifdef STEFANS_SENSORS_STATE
    const int nActions = 2, nStates = 16;
  #else
    const int nActions = 2, nStates = 10;
  #endif
  const unsigned maxLearnStepPerSim = 200; // random number... TODO

  comm->set_state_action_dims(nStates, nActions);
  // Tell smarties that action space should be bounded.
  // First action modifies curvature, only makes sense between -1 and 1
  // Second action affects Tp = (1+act[1])*Tperiod_0 (eg. halved if act[1]=-.5).
  // If too small Re=L^2*Tp/nu would increase too much, we allow it to
  //  double at most, therefore we set the bounds between -0.5 and 0.5.
  std::vector<double> upper_action_bound{1.,.25}, lower_action_bound{-1.,-.25};
  comm->set_action_scales(upper_action_bound, lower_action_bound, true);

  Simulation sim(argc, argv);
  sim.init();

  Shape * const object = sim.getShapes()[0];
  StefanFish*const agent = dynamic_cast<StefanFish*>( sim.getShapes()[1] );
  if(agent==nullptr) { printf("Agent was not a StefanFish!\n"); abort(); }

  if(comm->isTraining() == false) {
    sim.sim.verbose = true; sim.sim.muteAll = false;
    sim.sim.dumpTime = agent->Tperiod / 20;
  }
  unsigned sim_id = 0, tot_steps = 0;

  // Terminate loop if reached max number of time steps. Never terminate if 0
  while( true ) // train loop
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
    resetIC(agent, object, comm); // randomize initial conditions

    double t = 0, tNextAct = 0;
    unsigned step = 0;
    bool agentOver = false;

    comm->sendInitState( agent->state(object) ); //send initial state

    while (true) //simulation loop
    {
      setAction(agent, comm->recvAction(), tNextAct);
      tNextAct = getTimeToNextAct(agent, tNextAct);
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
      std::vector<double> state = agent->state(object);
      double reward = getReward(agent,object);

      if (agentOver || checkNaN(state, reward) ) {
        printf("Agent failed\n"); fflush(0);
        comm->sendTermState(state, reward);
        break;
      }
      else
      if (step >= maxLearnStepPerSim) {
        printf("Sim ended\n"); fflush(0);
        comm->sendLastState(state, reward);
        break;
      }
      else comm->sendState(state, reward);
    } // simulation is done

    if(comm->isTraining() == false) chdir("../");
    sim_id++;

    if (comm->terminateTraining()) return 0; // exit program
  }

  return 0;
}

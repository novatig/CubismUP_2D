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
#include "Communicator.h"
#include "Sim_FSI_Gravity.h"
#include "StefanFish.h"

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

inline void resetIC(
  StefanFish* const a, const Shape*const p, std::mt19937& gen) {
  uniform_real_distribution<Real> disA(-20./180*M_PI, 20./180*M_PI);
  uniform_real_distribution<Real> disX(-0.5, 0.5),  disY(-0.5, 0.5);
  Real C[2] = { p->centerOfMass[0] + (2+disX(gen))*a->length,
                p->centerOfMass[1] +    disY(gen) *a->length };
  a->setCenterOfMass(C);
  a->setOrientation(disA(gen));
}
inline void setAction(StefanFish* const agent,
  const vector<double> act, const double t) {
  agent->act(t, act);
}
inline vector<double> getState(
  const StefanFish* const a, const Shape*const p, const double t) {
  const double X = ( a->centerOfMass[0] - p->centerOfMass[0] )/ a->length;
  const double Y = ( a->centerOfMass[1] - p->centerOfMass[1] )/ a->length;
  const double A = a->getOrientation(), T = a->getPhase(t);
  const double U = a->getU() * a->Tperiod / a->length;
  const double V = a->getV() * a->Tperiod / a->length;
  const double W = a->getW() * a->Tperiod;
  const double lastT = a->lastTact, lastC = a->lastCurv, oldrC = a->oldrCurv;
  const vector<double> S = { X, Y, A, T, U, V, W, lastT, lastC, oldrC };
  printf("S:[%f %f %f %f %f %f %f %f %f %f]\n",X,Y,A,T,U,V,W,lastT,lastC,oldrC);
  return S;
}
inline bool isTerminal(const StefanFish*const a, const Shape*const p) {
  const double X = ( a->centerOfMass[0] - p->centerOfMass[0] )/ a->length;
  const double Y = ( a->centerOfMass[1] - p->centerOfMass[1] )/ a->length;
  assert(X>0);
  return std::fabs(Y)>1 || X<1 || X>3;
}
inline double getReward(const StefanFish* const a, const Shape*const p) {
  return isTerminal(a, p)? -1 : a->EffPDefBnd;
}
inline double getTimeToNextAct(const StefanFish* const agent, const double t) {
  return t + agent->getLearnTPeriod() / 2;
}

int app_main(
  Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv,    // arguments read from app's runtime settings file
  const unsigned numSteps  // number of time steps to run before exit
) {
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
  cout<<endl;
  cout<<endl; fflush(0);
  ArgumentParser parser(argc,argv);
  parser.set_strict_mode();
  const int nActions = 2;
  const int nStates = 10;
  const unsigned maxLearnStepPerSim = 200; // random number... TODO
  //const unsigned maxLearnStepPerSim = 1; // random number... TODO

  Communicator& communicator = *comm;
  communicator.update_state_action_dims(nStates, nActions);
  // Tell smarties that action space should be bounded.
  // First action modifies curvature, only makes sense between -1 and 1
  // Second action affects Tp = (1+act[1])*Tperiod_0 (eg. halved if act[1]=-.5).
  // If too small Re=L^2*Tp/nu would increase too much, we allow it to
  //  double at most, therefore we set the bounds between -0.5 and 0.5.
  vector<double> upper_action_bound{0.75, 0.25}, lower_action_bound{-0.75, -0.25};
  communicator.set_action_scales(upper_action_bound, lower_action_bound, true);

  Sim_FSI_Gravity sim(argc, argv);
  sim.init();

  Shape * const object = sim.getShapes()[0];
  StefanFish*const agent = dynamic_cast<StefanFish*>( sim.getShapes()[1] );
  if(agent==nullptr) { printf("Agent was not a StefanFish!\n"); abort(); }
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
    resetIC(agent, object, communicator.gen); // randomize initial conditions

    double t = 0, tNextAct = 0;
    unsigned step = 0;
    bool agentOver = false;

    communicator.sendInitState(getState(agent,object,t)); //send initial state

    while (true) //simulation loop
    {
      setAction(agent, communicator.recvAction(), tNextAct);
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
      const vector<double> state = getState(agent,object,t);
      const double reward = getReward(agent,object);

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
    sim.reset();
    chdir("../");
    sim_id++;
  }

  return 0;
}

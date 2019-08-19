//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include <unistd.h>
#include <sys/stat.h>

#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/BlowFish.h"

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

inline void resetIC(BlowFish* const agent, smarties::Communicator*const c)
{
  const Real A = 5*M_PI/180; // start between -5 and 5 degrees
  std::uniform_real_distribution<Real> dis(-A, A);
  const auto SA = c->isTraining() ? dis(c->getPRNG()) : 0.00;
  agent->setOrientation(SA);
}

inline void setAction(BlowFish* const agent, const std::vector<double> act)
{
  agent->flapAcc_R = act[0]/agent->timescale/agent->timescale;
  agent->flapAcc_L = act[1]/agent->timescale/agent->timescale;
}

inline std::vector<double> getState(const BlowFish* const agent)
{
  const double velscale = agent->radius / agent->timescale;
  const double w = agent->omega * agent->timescale;
  const double angle = agent->orientation;
  const double u = agent->u / velscale;
  const double v = agent->v / velscale;
  const double cosAng = std::cos(angle), sinAng = std::sin(angle);
  const double U = u*cosAng + v*sinAng, V = v*cosAng - u*sinAng;
  const double WR = agent->flapVel_R * agent->timescale;
  const double WL = agent->flapVel_L * agent->timescale;
  const double AR = agent->flapAng_R, AL = agent->flapAng_L;
  std::vector<double> states = {U, V, w, angle, AR, AL, WR, WL};
  printf("Sending [%f %f %f %f %f %f %f %f]\n", U,V,w,angle,AR,AL,WR,WL);
  return states;
}

inline double getReward(const BlowFish* const agent)
{
  const double velscale = agent->radius / agent->timescale;
  const double u = agent->u / velscale, v = agent->v / velscale;
  const double cosAng = std::cos(agent->orientation);
  const bool ended = cosAng<0;
  const double reward = ended ? -10 : cosAng -std::sqrt(u*u+v*v);
  return reward;
}

inline bool isTerminal(const BlowFish* const agent)
{
  return std::cos(agent->orientation)<0;
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

inline double getTimeToNextAct(const BlowFish* const agent, const double t)
{
  return t + agent->timescale / 4;
}

inline void app_main(
  smarties::Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,                  // mpi_comm that mpi-based apps can use
  int argc, char**argv             // args read from app's runtime settings file
) {
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
  const int nActions = 2, nStates = 8;
  const unsigned maxLearnStepPerSim = comm->isTraining()? 500
                                     : std::numeric_limits<int>::max();

  comm->set_state_action_dims(nStates, nActions);

  Simulation sim(argc, argv);
  sim.init();

  BlowFish* const agent = dynamic_cast<BlowFish*>( sim.getShapes()[0] );
  if(agent==nullptr) { printf("Obstacle was not a BlowFish!\n"); abort(); }
  if(comm->isTraining() == false) {
    sim.sim.verbose = true; sim.sim.muteAll = false;
    sim.sim.dumpTime = agent->timescale / 20;
  }
  unsigned sim_id = 0, tot_steps = 0;

  // Terminate loop if reached max number of time steps. Never terminate if 0
  while( 1 ) // train loop
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

    comm->sendInitState(getState(agent)); //send initial state

    while (true) //simulation loop
    {
      setAction(agent, comm->recvAction());
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
      std::vector<double> state = getState(agent);
      double reward = getReward(agent);

      if ( agentOver || checkNaN(state, reward) ) {
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

    if (comm->terminateTraining()) return; // exit program
  }
}


int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}

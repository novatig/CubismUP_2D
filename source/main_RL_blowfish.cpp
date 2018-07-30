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

//
// All these functions are defined here and not in object itself because
// Many different tasks, each requiring different state/act/rew descriptors
// could be designed for the same objects (also to fully encapsulate RL).
//

inline void resetIC(Blowfish* const agent, std::mt19937& gen) {
  agent->setOrientation(0);
}
inline void setAction(Blowfish* const agent, const vector<double> act) {
  agent->flapAcc_R = act[0]/agent->timescale/agent->timescale;
  agent->flapAcc_L = act[1]/agent->timescale/agent->timescale;
}
inline vector<double> getState(const Blowfish* const agent) {
  const double w = agent->getW() * agent->timescale;
  const double angle = agent->getOrientation();
  const double u = agent->getU() / agent->velscale;
  const double v = agent->getV() / agent->velscale;
  const double cosAng = std::cos(angle), sinAng = std::sin(angle);
  const double U = u*cosAng + v*sinAng, V = v*cosAng - u*sinAng;
  const double WR = agent->flapVel_R * agent->timescale;
  const double WL = agent->flapVel_L * agent->timescale;
  const double AR = agent->flapAng_R, AL = agent->flapAng_L;
  vector<double> states = {U, V, w, angle, AR, AL, WR, WL};
  printf("Sending [%f %f %f %f %f %f %f %f]\n", U,V,w,angle,AR,AL,WR,WL);
  return states;
}
inline double getReward(const Blowfish* const agent) {
  const double angle = agent->getOrientation();
  const double u = agent->getU() / agent->velscale;
  const double v = agent->getV() / agent->velscale;
  const double cosAng = std::cos(angle);
  const bool ended = cosAng<0;
  const double reward = ended ? -10 : cosAng -std::sqrt(u*u+v*v);
  return reward;
}
inline bool isTerminal(const Blowfish* const agent) {
  const Real angle = agent->getOrientation();
  const Real cosAng = std::cos(angle);
  return cosAng<0;
}
inline double getTimeToNextAct(const Blowfish* const agent, const double t) {
  return agent->timescale * 0.1;
}

int main(int argc, char **argv)
{
	ArgumentParser parser(argc,argv);
	parser.set_strict_mode();
	const int socket_id  = parser("-Socket").asInt(-1);
  printf("Starting communication with RL over socket %d\n", socket_id);

  const int nActions = 2;
  const int nStates = 8;
  Communicator communicator(socket_id, nStates, nActions);

	char dirname[1024]; dirname[1023] = '\0';
  unsigned sim_id = 0;

	while(true) // train loop
	{
		sprintf(dirname, "run_%u/", sim_id);
		printf("Starting a new sim in directory %s\n", dirname);
		mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		chdir(dirname);

		Sim_FSI_Gravity sim(argc, argv);
		sim.init();

    Blowfish* const agent = dynamic_cast<Blowfish*>( sim.getShape() );
    if(agent==nullptr) { printf("Obstacle was not a Blowfish!\n"); abort(); }
    resetIC(agent, communicator.gen); // randomize initial conditions

    communicator.sendInitState(getState(agent)); //send initial state

    Real t = 0;
    bool simIsOver = false;
    bool agentOver = false;

    while (true) //simulation loop
    {
      setAction(agent, communicator.recvAction());
      const double tNextAct = getTimeToNextAct(agent, t);
      while (t < tNextAct)
      {
        const double dt = sim.calcMaxTimestep();
        if ( sim.advance( dt ) ) { simIsOver = true; }
        if ( isTerminal(agent) ) { agentOver = true; }
        if ( simIsOver || agentOver )  break;
      }

      const vector<double> state = getState(agent);
      const double reward = getReward(agent);

      if (agentOver) {
        communicator.sendTermState(state, reward);
        break;
      }
      else
      if (simIsOver) {
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

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Glider.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"

std::vector<double> Glider::state() const
{
  const Real cosAng = std::cos(orientation), sinAng = std::sin(orientation);
  //Nondimensionalization:
  const Real xdot = u/velscale, ydot = v/velscale, L = lengthscale;
  const Real X = labCenterOfMass[0]/L, Y = labCenterOfMass[1]/L;
  const Real U = xdot*cosAng +ydot*sinAng;
  const Real V = ydot*cosAng -xdot*sinAng;
  const Real W = omega*timescale;
  const Real T = appliedTorque/torquescale;
  std::vector<double> state = {U, V, W, X, Y, cosAng, sinAng, T, xdot, ydot};
  printf("Sending (%lu) [%f %f %f %f %f %f %f %f %f %f]\n",
    state.size(),U,V,W,X,Y,cosAng,sinAng,T,xdot,ydot);
  return state;
}

double Glider::reward()
{
  const double DT = DTactions_nonDim, L = lengthscale;
  const double X = labCenterOfMass[0]/L;//, Y = labCenterOfMass[1]/L;
  //const double cosAng = std::cos(orientation), sinAng = std::sin(orientation);
  const double dist_gain = old_distance - std::fabs(X-100);
  //const double rotation = std::fabs(old_angle - std::atan2(sinAng,cosAng))/DT;
  //const double jerk = std::fabs(old_torque - appliedTorque)/DT/torquescale;
  const double performamce = std::pow(appliedTorque/torquescale, 2);
  return rewardType==1 ? dist_gain - DT : dist_gain - performamce;
}

double Glider::terminalReward()
{
  const double cosAng = std::cos(orientation), sinAng = std::sin(orientation);
  const double angle = std::atan2(sinAng,cosAng), L = lengthscale;
  const double X = labCenterOfMass[0]/L;//, Y = labCenterOfMass[1]/L;
  //these rewards will then be multiplied by 1/(1-gamma)
  //in RL algorithm, so that internal RL scales make sense
  const double dist = std::fabs(X - 100), rela = std::fabs(angle - M_PI/4);
  const double xrew = dist>5 ? 0 : std::exp(-dist*dist);
  const double arew = (rela>M_PI/4 || dist>5) ? 0 : std::exp(-10*rela*rela);
  double final_reward  = termRew_fac*(xrew + arew);

  #ifdef SPEED_PENAL
  {
    if (std::fabs(_s.u) > 0.5)
      final_reward *= std::exp(-10*std::pow(std::fabs(_s.u)-.5,2));
    if (std::fabs(_s.v) > 0.5)
      final_reward *= std::exp(-10*std::pow(std::fabs(_s.v)-.5,2));
    if (std::fabs(_s.w) > 0.5)
      final_reward *= std::exp(-10*std::pow(std::fabs(_s.w)-.5,2));
  }
  #endif

  return final_reward - dist;
}

bool Glider::isOver()
{
  const Real X = labCenterOfMass[0]/lengthscale, Y = labCenterOfMass[1]/lengthscale;
  const bool way_too_far = X > 150;
  const double slack = 0.4 * std::max((Real)0, std::min(X-50, 100-X));
  const bool hit_bottom =  Y <= -50 -slack;
  const bool wrong_xdir = X < -50;
  const bool timeover = sim.time / timescale > 1000;
  return ( timeover || hit_bottom || wrong_xdir || way_too_far );
}

void Glider::act(std::vector<double> action)
{
  const Real X = labCenterOfMass[0]/lengthscale;
  const Real cosAng = std::cos(orientation), sinAng = std::sin(orientation);
  old_distance = std::fabs(X - 100);
  old_torque = appliedTorque;
  old_angle = std::atan2(sinAng, cosAng);
  printf("Received action [%f]\n", action[0]);
  appliedTorque = action[0]*torquescale;
}

void Glider::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Ellipse K(semiAxis[0],semiAxis[1], h, center, orientation, rhoS);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
      if(K.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        //obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        K(vInfo[i], b, * obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

void Glider::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);

  energy -= std::pow(appliedTorque/torquescale, 2) * dt;
  //powerOutput += dt*Torque*Torque;
}

void Glider::computeForces()
{
  Shape::computeForces();
  //energySurf += PoutBnd * sim.dt;
  //energySurf += Pthrust/(Pthrust-std::min(Pout,(double)0)) * sim.dt;
  //energySurf += Pthrust/(Pthrust-std::min(PoutBnd,(double)0)) * sim.dt;
}

Glider::Glider(SimulationData& s, ArgumentParser& p, double C[2] ) :
  Shape(s,p,C), rewardType(p("-rewType").asInt(1)),
  semiAxis{(Real) p("-semiAxisX").asDouble(),
           (Real) p("-semiAxisY").asDouble() }
{
  printf("Created Glider with axes:%f %f rho:%f\n",semiAxis[0],semiAxis[1], rhoS);
}

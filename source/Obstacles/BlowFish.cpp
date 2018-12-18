//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "BlowFish.h"
#include "ShapeLibrary.h"

BlowFish::BlowFish(SimulationData&s, ArgumentParser&p, double C[2])
: Shape(s,p,C), radius( p("-radius").asDouble(0.1) )
{
  const double distHalfCM = 4*radius/(3*M_PI);
  const double halfarea = 0.5*M_PI*radius*radius;
  // based on weighted average
  const double CentTop =  distHalfCM;
  const double MassTop =  halfarea*rhoTop;
  const double CentBot = -distHalfCM;
  const double MassBot =  halfarea*rhoBot;
  const double CentFin = -std::sin(finAngle0)*(attachDist+finLength/2);
  const double MassFin = 2*finLength*finWidth*rhoFin;

  d_gm[0] = 0;
  d_gm[1] = -(CentTop*MassTop + CentBot*MassBot + CentFin*MassFin)/(MassTop + MassBot + MassFin);

  centerOfMass[0]=center[0]-std::cos(orientation)*d_gm[0]+std::sin(orientation)*d_gm[1];
  centerOfMass[1]=center[1]-std::sin(orientation)*d_gm[0]-std::cos(orientation)*d_gm[1];
  //cout << "Created BlowFish"
}

void BlowFish::resetAll() {
  flapAng_R = 0; flapAng_L = 0;
  flapVel_R = 0; flapVel_L = 0;
  flapAcc_R = 0; flapAcc_L = 0;
  Shape::resetAll();
}

void BlowFish::updatePosition(double dt)
{
  Shape::updatePosition(dt);

  flapAng_R += dt*flapVel_R + .5*dt*dt*flapAcc_R;
  flapAng_L += dt*flapVel_L + .5*dt*dt*flapAcc_L;
  flapVel_R += dt*flapAcc_R;
  flapVel_L += dt*flapAcc_L;

  if(flapAng_R > M_PI/2) { //maximum extent of fin is pi/2
    printf("Blocked flapAng_R at  M_PI/2\n");
    flapAng_R =  M_PI/2;
    if(flapVel_R>0) flapVel_R = 0;
    if(flapAcc_R>0) flapAcc_R = 0;
  }
  if(flapAng_R < -M_PI/2) { //maximum extent of fin is pi/2
    printf("Blocked flapAng_R at -M_PI/2\n");
    flapAng_R = -M_PI/2;
    if(flapVel_R<0) flapVel_R = 0;
    if(flapAcc_R<0) flapAcc_R = 0;
  }
  if(flapAng_L > M_PI/2) { //maximum extent of fin is pi/2
    printf("Blocked flapAng_L at  M_PI/2\n");
    flapAng_L =  M_PI/2;
    if(flapVel_L>0) flapVel_L = 0;
    if(flapAcc_L>0) flapAcc_L = 0;
  }
  if(flapAng_L < -M_PI/2) { //maximum extent of fin is pi/2
    printf("Blocked flapAng_L at -M_PI/2\n");
    flapAng_L = -M_PI/2;
    if(flapVel_L<0) flapVel_L = 0;
    if(flapAcc_L<0) flapAcc_L = 0;
  }

  #ifndef RL_TRAIN
  if(sim.verbose)
  if(  std::fabs(flapVel_R) + std::fabs(flapAcc_R) + std::fabs(flapVel_L)
     + std::fabs(flapAcc_L) > std::numeric_limits<Real>::epsilon() )
    printf("[ang, angvel, angacc] : right:[%f %f %f] left:[%f %f %f]\n",
      flapAng_R, flapVel_R, flapAcc_R, flapAng_L, flapVel_L, flapAcc_L);
  #endif
}

void BlowFish::create(const vector<BlockInfo>& vInfo)
{
  const double angleAttFin1 = orientation  -finAngle0;
  const double angleAttFin2 = orientation  +finAngle0 +M_PI;
  const double angleTotFin1 = angleAttFin1 +flapAng_R;
  const double angleTotFin2 = angleAttFin2 +flapAng_L;

  const double attach1[2] = {
    center[0]+attachDist*std::cos(angleAttFin1),
    center[1]+attachDist*std::sin(angleAttFin1)
  };
  const double attach2[2] = {
    center[0]+attachDist*std::cos(angleAttFin2),
    center[1]+attachDist*std::sin(angleAttFin2)
  };

  const Real h = vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    const FillBlocks_VarRhoCylinder kernelC(radius, h, center, rhoTop, rhoBot, orientation);
    const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, angleTotFin1, flapVel_R, rhoFin);
    const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, angleTotFin2, flapVel_L, rhoFin);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      if(kernelC._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
      else if(kernelF1._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
      else if(kernelF2._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }

      const auto pos = obstacleBlocks[vInfo[i].blockID];
      if(pos == nullptr) continue;

      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      kernelC(vInfo[i], b, pos);
      kernelF1(vInfo[i], b, pos);
      kernelF2(vInfo[i], b, pos);
    }
  }

  removeMoments(vInfo);
  for (auto & O : obstacleBlocks) if(O not_eq nullptr) O->allocate_surface();
}

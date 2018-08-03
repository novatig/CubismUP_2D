//
//  Shape.h
//  CubismUP_2D
//
//  Virtual shape class which defines the interface
//  Default simple geometries are also provided and can be used as references
//
//  This class only contains static information (position, orientation,...), no dynamics are included (e.g. velocities,...)
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "BlowFish.h"
#include "ShapeLibrary.h"

BlowFish::BlowFish(SimulationData&s, ArgumentParser&p, Real C[2])
: Shape(s,p,C), radius( p("-radius").asDouble(0.1) )
{
  const Real distHalfCM = 4*radius/(3*M_PI);
  const Real halfarea = 0.5*M_PI*radius*radius;
  // based on weighted average
  const Real CentTop =  distHalfCM;
  const Real MassTop =  halfarea*rhoTop;
  const Real CentBot = -distHalfCM;
  const Real MassBot =  halfarea*rhoBot;
  const Real CentFin = -std::sin(finAngle0)*(attachDist+finLength/2);
  const Real MassFin = 2*finLength*finWidth*rhoFin;

  d_gm[0] = 0;
  d_gm[1] = -(CentTop*MassTop + CentBot*MassBot + CentFin*MassFin)/(MassTop + MassBot + MassFin);

  centerOfMass[0]=center[0]-std::cos(orientation)*d_gm[0]+std::sin(orientation)*d_gm[1];
  centerOfMass[1]=center[1]-std::sin(orientation)*d_gm[0]-std::cos(orientation)*d_gm[1];
  //cout << "Created BlowFish"
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
  const Real angleAttFin1 = orientation  -finAngle0;
  const Real angleAttFin2 = orientation  +finAngle0 +M_PI;
  const Real angleTotFin1 = angleAttFin1 +flapAng_R;
  const Real angleTotFin2 = angleAttFin2 +flapAng_L;

  const Real attach1[2] = {
    center[0]+attachDist*std::cos(angleAttFin1),
    center[1]+attachDist*std::sin(angleAttFin1)
  };
  const Real attach2[2] = {
    center[0]+attachDist*std::cos(angleAttFin2),
    center[1]+attachDist*std::sin(angleAttFin2)
  };

  const Real h = vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();

  {
    const FillBlocks_Cylinder kernelC(radius, h, center, rhoBot);
    const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, angleTotFin1, flapVel_R, rhoFin);
    const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, angleTotFin2, flapVel_L, rhoFin);

    for(size_t i=0; i<vInfo.size(); i++) {
      const BlockInfo& info = vInfo[i];
      if(kernelC._is_touching(info))
      {
        assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
        obstacleBlocks[info.blockID] = new ObstacleBlock;
        obstacleBlocks[info.blockID]->clear(); //memset 0
      }
      else if(kernelF1._is_touching(info))
      {
        assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
        obstacleBlocks[info.blockID] = new ObstacleBlock;
        obstacleBlocks[info.blockID]->clear(); //memset 0
      }
      else if(kernelF2._is_touching(info))
      {
        assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
        obstacleBlocks[info.blockID] = new ObstacleBlock;
        obstacleBlocks[info.blockID]->clear(); //memset 0
      }
    }
  }

  #pragma omp parallel
  {
    const FillBlocks_VarRhoCylinder kernelC(radius, h, center, rhoTop, rhoBot, orientation);
    const FillBlocks_Plate kernelF1(finLength, finWidth, h, attach1, angleTotFin1, flapVel_R, rhoFin);
    const FillBlocks_Plate kernelF2(finLength, finWidth, h, attach2, angleTotFin2, flapVel_L, rhoFin);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      const BlockInfo& info = vInfo[i];
      const auto pos = obstacleBlocks.find(info.blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      kernelC(info, b, pos->second);
      kernelF1(info, b, pos->second);
      kernelF2(info, b, pos->second);
    }
  }

  removeMoments(vInfo);
  for (auto & block : obstacleBlocks) block.second->allocate_surface();
}

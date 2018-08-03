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


#include "ShapeLibrary.h"
#include "ShapesSimple.h"

void Disk::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();

  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    for(size_t i=0; i<vInfo.size(); i++) {
      if(kernel._is_touching(vInfo[i])) { //position of sphere + radius + 2*h safety
        assert(obstacleBlocks.find(vInfo[i].blockID) == obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }
  }

  #pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      kernel(vInfo[i], b, pos->second);
    }
  }

  removeMoments(vInfo);
  for (auto & block : obstacleBlocks) block.second->allocate_surface();
}

void HalfDisk::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();

  {
    FillBlocks_HalfCylinder kernel(radius, h, center, rhoS, orientation);
    for(size_t i=0; i<vInfo.size(); i++) {
      if(kernel._is_touching(vInfo[i])) { //position of sphere + radius + 2*h safety
        assert(obstacleBlocks.find(vInfo[i].blockID) == obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }
  }

  #pragma omp parallel
  {
    FillBlocks_HalfCylinder kernel(radius, h, center, rhoS, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      kernel(vInfo[i], b, pos->second);
    }
  }

  removeMoments(vInfo);
  for (auto & block : obstacleBlocks) block.second->allocate_surface();
}

void Ellipse::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();

  {
    FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);
    for(size_t i=0; i<vInfo.size(); i++) {
      //const auto pos = obstacleBlocks.find(info.blockID);
      if(kernel._is_touching(vInfo[i])) { //position of sphere + radius + 2*h safety
        assert(obstacleBlocks.find(vInfo[i].blockID)==obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }
  }
  #pragma omp parallel
  {
    FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      kernel(vInfo[i], b, pos->second);
    }
  }

  const FillBlocks_EllipseFinalize finalize(h, rhoS);
  compute(finalize, vInfo);

  for (auto & block : obstacleBlocks) block.second->allocate_surface();
}

void DiskVarDensity::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();
  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    for(size_t i=0; i<vInfo.size(); i++) {
      if(kernel._is_touching(vInfo[i])) {
        assert(obstacleBlocks.find(vInfo[i].blockID) == obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }
  }
  #pragma omp parallel
  {
    FillBlocks_VarRhoCylinder kernel(radius, h, center, rhoTop, rhoBot, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      BlockInfo info = vInfo[i];
      const auto pos = obstacleBlocks.find(info.blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      kernel(info, b, pos->second);
    }
  }

  removeMoments(vInfo);
  for (auto & block : obstacleBlocks) block.second->allocate_surface();
}

void EllipseVarDensity::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();
  {
    FillBlocks_Ellipse kernel(semiAxisX, semiAxisY, h, center, orientation, rhoTop);
    for(size_t i=0; i<vInfo.size(); i++) {
      if(kernel._is_touching(vInfo[i])) {
        assert(obstacleBlocks.find(vInfo[i].blockID) == obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }
  }
  #pragma omp parallel
  {
    FillBlocks_Ellipse kernel(semiAxisX, semiAxisY, h, center, orientation, rhoTop);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      BlockInfo info = vInfo[i];
      const auto pos = obstacleBlocks.find(info.blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      kernel(info, b, pos->second);
    }
  }

  const FillBlocks_VarRhoEllipseFinalize finalize(h, center, rhoTop, rhoBot, orientation);
  compute(finalize, vInfo);
  removeMoments(vInfo);
  for (auto & block : obstacleBlocks) block.second->allocate_surface();
}

#if 0 //def RL_TRAIN
#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10
void Ellipse::act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) override
{
  assert(time_ptr not_eq nullptr);
  assert(communicator not_eq nullptr);
  if(!initialized_time_next_comm || *time_ptr>time_next_comm)
  {
    const Real w = *omegaBody, u = *uBody, v = *vBody;
    const Real cosAng = cos(orientation), sinAng = sin(orientation);
    const Real angle = atan2(sinAng,cosAng);

    //Nondimensionalization:
    const Real xdot = u/velscale, ydot = v/velscale;
    const Real X = labCenterOfMass[0]/a, Y = labCenterOfMass[1]/a;
    const Real U = xdot*cosAng +ydot*sinAng;
    const Real V = ydot*cosAng -xdot*sinAng;
    const Real W = w*timescale;
    const Real T = Torque/torquescale;
    const bool ended = X>125 || X<-10 || Y<=-50;
    const bool landing = std::fabs(angle - .25*M_PI) < 0.1;
    const Real vertDist = std::fabs(Y+50), horzDist = std::fabs(X-100);

    Real reward;
    if (ended)
    {
      info = _AGENT_LASTCOMM;
      reward= (X>125 || X<-10) ? -100 -HEIGHT_PENAL*vertDist
            : (horzDist<1? (landing?2:1) * TERM_REW_FAC : -horzDist) ;
    } else
      reward = (old_Dist-horzDist) -fabs(Torque-old_Torque)/.5;
    //-(powerOutput-old_powerOutput);

    vector<double> state = {U, V, W, X, Y, cosAng, sinAng, T, xdot, ydot}; vector<double> action = {0.};

    printf("Sending (%lu) [%f %f %f %f %f %f %f %f %f %f], %f %f\n",
    state.size(),U,V,W,X,Y,cosAng,sinAng,T,xdot,ydot, Torque,torquescale);

    communicator->sendState(0, info, state, reward);

    if(info == _AGENT_LASTCOMM) abort();
    old_Dist = horzDist;
    old_Torque = Torque;
    old_powerOutput = powerOutput;
    initialized_time_next_comm = true;
    time_next_comm = time_next_comm + 0.5*timescale;
    info = _AGENT_NORMALCOMM;

    communicator->recvAction(action);
       printf("Received %f\n", action[0]);
    Torque = action[0]*torquescale;
  }

  *omegaBody += dt*Torque/J;
  powerOutput += dt*Torque*Torque;
}
#endif

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


#include "SmartCylinder.h"
#include "ShapeLibrary.h"
#include "BufferedLogger.h"

std::vector<double> SmartCylinder::state(const Real OX, const Real OY, const Real velScale) const
{
  const std::vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  // relative position 2
  // relative velocity (+ angular) 3
  // shear velocity sensors 8
  std::vector<double> ret (5 + 8, 0);
  const Real timeScale = 2*radius / velScale;
  ret[0] = (center[0] - OX)/(2*radius);
  ret[1] = (center[1] - OY)/(2*radius);
  ret[2] = u / velScale;
  ret[3] = v / velScale;
  ret[4] = omega * timeScale;

  const auto holdsPoint = [&](const BlockInfo&info, const Real X, const Real Y)
  {
    Real MIN[2], MAX[2];
    info.pos(MIN, 0,0);
    info.pos(MAX, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    for(int i=0; i<2; ++i) {
        MIN[i] -= 0.5 * info.h_gridpoint; // pos returns cell centers
        MAX[i] += 0.5 * info.h_gridpoint; // we care about whole block
    }
    return X >= MIN[0] && Y >= MIN[1] && X <= MAX[0] && Y <= MAX[1];
  };

  //#pragma omp parallel for
  for(int a = 0; a<8; a++)
  {
    const Real theta = a * 2 * M_PI / 8, dist = 1.1 * radius;
    const Real cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    const Real sensorX = center[0] - dist * std::cos(theta);
    const Real sensorY = center[1] + dist * std::sin(theta);
    for(size_t i=0; i<vInfo.size(); i++)
    {
      if(not holdsPoint(vInfo[i], sensorX, sensorY) ) continue;
      Real org[2]; vInfo[i].pos(org, 0, 0);
      const Real invh = 1 / vInfo[i].h_gridpoint;
      const int indx = (int) std::round((sensorX - org[0])*invh);
      const int indy = (int) std::round((sensorY - org[1])*invh);
      const int clipIndX = std::min( std::max(0, indx), FluidBlock::sizeX-1);
      const int clipIndY = std::min( std::max(0, indy), FluidBlock::sizeY-1);
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      const Real VX = b(clipIndX, clipIndY).u-u, VY = b(clipIndX, clipIndY).v-v;
      ret[5 + a] = (cosTheta*VY + sinTheta*VX) / velScale;
      Real pos[2]; vInfo[i].pos(pos, clipIndX, clipIndY);
      //printf("sensor:[%f %f]->[%f %f] ind:[%d %d] val:%f\n",
      //sensorX, sensorY, pos[0], pos[1], indx, indy, ret[5 + a]);
      break;
    }
  }

  return ret;
}

void SmartCylinder::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0

        const auto pos = obstacleBlocks[vInfo[i].blockID];
        assert(pos not_eq nullptr);

        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos);
      }
  }

  removeMoments(vInfo);
  for (auto & o : obstacleBlocks) if(o not_eq nullptr) o->allocate_surface();
}

void SmartCylinder::computeVelocities()
{
  const std::vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  double _M = 0, _J = 0, FFX = 0, FFY = 0, FTZ = 0; //linear momenta
  const double lamdahsq = std::pow(vInfo[0].h_gridpoint,2) * sim.lambda;

  #pragma omp parallel for schedule(dynamic) reduction(+:_M,_J,FFX,FFY,FTZ)
  //,CX,CY)
  for(size_t i=0; i<vInfo.size(); i++)
  {
      const auto pos = obstacleBlocks[vInfo[i].blockID];
      if(pos == nullptr) continue;

      const Real (&CHI)[32][32] = pos->chi;
      //const Real (&RHO)[32][32] = pos->rho;
      FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        const Real chi = CHI[iy][ix];
        if (chi <= 0) continue;
        double p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        const double uDiff = b(ix,iy).u - (u -omega*p[1]);
        const double vDiff = b(ix,iy).v - (v +omega*p[0]);
        const double rhochi = 1 * chi * lamdahsq; //RHO[iy][ix]
        _M  += rhochi;
        FFX += rhochi * uDiff;//(b(ix,iy).u - u);
        FFY += rhochi * vDiff;//(b(ix,iy).v - v);
        //CX += rhochi * p[0]; CY += rhochi * p[1];
        _J  += rhochi * (p[0]*p[0]  + p[1]*p[1]);
        FTZ += rhochi * (p[0]*vDiff - p[1]*uDiff );
      }
  }
  //cout << CX << " " << CY << endl;
  //const Real forceScale = 0.1*0.1 * 2*radius;
  //appliedForceX = -2 * forceScale;
  //appliedForceY = 0 * forceScale;
  //appliedTorque = 0 * forceScale * radius;
  //cout << appliedForceX << " " << FFX << endl;
  const Real accx = ( appliedForceX + FFX ) / _M;
  const Real accy = ( appliedForceY + FFY ) / _M;
  const Real acca = ( appliedTorque + FTZ ) / _J;

  u = u + sim.dt * accx;
  v = v + sim.dt * accy;
  omega = omega + sim.dt * acca;
  energy -= ( std::pow(appliedForceX, 2) + std::pow(appliedForceY, 2)
            + std::pow(appliedTorque /(2*radius), 2) ) * sim.dt;

  #ifndef RL_TRAIN
    if(sim.verbose)
      printf("CM:[%f %f] C:[%f %f] u:%f v:%f omega:%f M:%f J:%f V:%f\n",
      centerOfMass[0],centerOfMass[1], center[0],center[1], u,v,omega, M, J, V);
    if(not sim.muteAll)
    {
      stringstream ssF;
      ssF<<sim.path2file<<"/velocity_"<<obstacleID<<".dat";
      std::stringstream &fileSpeed = logger.get_stream(ssF.str());
      if(sim.step==0)
        fileSpeed<<"time dt CMx CMy angle u v omega M J FFX FFY FTZ"<<std::endl;

      fileSpeed<<sim.time<<" "<<sim.dt<<" "<<centerOfMass[0]<<" "<<centerOfMass[1]<<" "<<orientation<<" "<<u <<" "<<v<<" "<<omega<<" "<<M<<" "<<J<<" "<<FFX<<" "<<FFY<<" "<<FTZ<<endl;
    }
  #endif

  if(bForcedx) { u = forcedu; cout << "bForcedx" << endl; }
  if(bForcedy) v = forcedv;
  if(bBlockang) omega = 0;
}

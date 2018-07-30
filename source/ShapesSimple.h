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

#pragma once

#include "ShapeLibrary.h"
#include "Shape.h"
#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10

class Disk : public Shape
{
  const Real radius;
 public:
  Disk(SimulationData& s, ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) {
    printf("Created a Disk with: R:%f rho:%f\n",radius,rhoS);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }

  void outputSettings(ostream &outStream) const override
  {
    outStream << "Disk\n";
    outStream << "radius " << radius << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override
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
};

class HalfDisk : public Shape
{
 protected:
  const Real radius;

 public:
  HalfDisk( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) {
    printf("Created a half Disk with: R:%f rho:%f\n",radius,rhoS);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }

  void outputSettings(ostream &outStream) const override
  {
    outStream << "HalfDisk\n";
    outStream << "radius " << radius << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override
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
};

class Ellipse : public Shape
{
 protected:
  const Real semiAxis[2];
  //Characteristic scales:
  const Real majax = std::max(semiAxis[0], semiAxis[1]);
  const Real minax = std::min(semiAxis[0], semiAxis[1]);
  const Real velscale = std::sqrt((rhoS/1-1)*9.81*minax);
  const Real lengthscale = majax, timescale = majax/velscale;
  //const Real torquescale = M_PI/8*pow((a*a-b*b)*velscale,2)/a/b;
  const Real torquescale = M_PI*majax*majax*velscale*velscale;

  Real Torque = 0, old_Torque = 0, old_Dist = 100;
  Real powerOutput = 0, old_powerOutput = 0;

 public:
  Ellipse(SimulationData&s, ArgumentParser&p, Real C[2]) : Shape(s,p,C),
    semiAxis{ (Real) p("-semiAxisX").asDouble(.1),
              (Real) p("-semiAxisY").asDouble(.2) } {
    printf("Created ellipse semiAxis:[%f %f] rhoS:%f a:%f b:%f velscale:%f lengthscale:%f timescale:%f torquescale:%f\n", semiAxis[0], semiAxis[1], rhoS, majax, minax, velscale, lengthscale, timescale, torquescale); fflush(0);
  }

  Real getCharLength() const  override
  {
    return 2 * max(semiAxis[1],semiAxis[0]);
  }

  void outputSettings(ostream &outStream) const override
  {
    outStream << "Ellipse\n";
    outStream << "semiAxisX " << semiAxis[0] << endl;
    outStream << "semiAxisY " << semiAxis[1] << endl;

    Shape::outputSettings(outStream);
  }

  #if 0 //def RL_TRAIN
  void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) override
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

  void create(const vector<BlockInfo>& vInfo) override
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
};

class DiskVarDensity : public Shape
{
 protected:
  const Real radius;
  const Real rhoTop;
  const Real rhoBot;

 public:
  DiskVarDensity( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  rhoTop(p("-rhoTop").asDouble(rhoS) ), rhoBot(p("-rhoBot").asDouble(rhoS) ) {
    d_gm[0] = 0;
    // based on weighted average between the centers of mass of half-disks:
    d_gm[1] = -4.*radius/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

    centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
    centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
  }

  Real getCharLength() const  override
  {
    return 2 * radius;
  }

  void create(const vector<BlockInfo>& vInfo) override
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

  void outputSettings(ostream &outStream) const override
  {
    outStream << "DiskVarDensity\n";
    outStream << "radius " << radius << endl;
    outStream << "rhoTop " << rhoTop << endl;
    outStream << "rhoBot " << rhoBot << endl;

    Shape::outputSettings(outStream);
  }
};

class EllipseVarDensity : public Shape
{
  protected:
   const Real semiAxisX;
   const Real semiAxisY;
   const Real rhoTop;
   const Real rhoBot;

  public:
   EllipseVarDensity( SimulationData& s, ArgumentParser& p, Real C[2] ) :
   Shape(s,p,C),
   semiAxisX( p("-semiAxisX").asDouble(0.1) ),
   semiAxisY( p("-semiAxisY").asDouble(0.1) ),
   rhoTop(p("-rhoTop").asDouble(rhoS) ), rhoBot(p("-rhoBot").asDouble(rhoS) ) {
     d_gm[0] = 0;
     // based on weighted average between the centers of mass of half-disks:
     d_gm[1] = -4.*semiAxisY/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

     centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
     centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
   }

   Real getCharLength() const override {
     return 2 * std::max(semiAxisX, semiAxisY);
   }

   void create(const vector<BlockInfo>& vInfo) override
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

   void outputSettings(ostream &outStream) const override
   {
     outStream << "Ellipse\n";
     outStream << "semiAxisX " << semiAxisX << endl;
     outStream << "semiAxisY " << semiAxisY << endl;
     outStream << "rhoTop " << rhoTop << endl;
     outStream << "rhoBot " << rhoBot << endl;

     Shape::outputSettings(outStream);
   }
};

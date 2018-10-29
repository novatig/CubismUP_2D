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

#include "common.h"
#include "Definitions.h"
#include "ObstacleBlock.h"

class Shape
{
 public: // data fields
  SimulationData& sim;
  unsigned obstacleID = 0;
  std::vector<ObstacleBlock*> obstacleBlocks;
  // general quantities
  const double origC[2], origAng;
  double center[2]; // for single density, this corresponds to centerOfMass
  double centerOfMass[2];
  double d_gm[2] = {0,0}; // distance of center of geometry to center of mass
  double labCenterOfMass[2] = {0,0};
  double orientation;
  double M = 0;
  double V = 0;
  double J = 0;
  double u = 0; // in lab frame, not sim frame
  double v = 0; // in lab frame, not sim frame
  double omega = 0;
  double computedu = 0;
  double computedv = 0;
  double computedo = 0;
  double area_penal = 0;
  double mass_penal = 0;
  double forcex_penal = 0;
  double forcey_penal = 0;
  double torque_penal = 0;

  double perimeter=0, forcex=0, forcey=0, forcex_P=0, forcey_P=0;
  double forcex_V=0, forcey_V=0, torque=0, torque_P=0, torque_V=0;
  double drag=0, thrust=0, circulation=0, Pout=0, PoutBnd=0, defPower=0;
  double defPowerBnd=0, Pthrust=0, Pdrag=0, EffPDef=0, EffPDefBnd=0;

  const double rhoS;
  const bool bForced;
  const bool bFixed;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const bool bFixedx;
  const bool bFixedy;
  const double forcedu;
  const double forcedv;

  virtual void resetAll() {
             center[0] = origC[0];
             center[1] = origC[1];
       centerOfMass[0] = origC[0];
       centerOfMass[1] = origC[1];
    labCenterOfMass[0] = 0;
    labCenterOfMass[1] = 0;
    orientation = origAng;
    M = 0;
    V = 0;
    J = 0;
    u = 0;
    v = 0;
    omega = 0;
    computedu = 0;
    computedv = 0;
    computedo = 0;
    d_gm[0] = 0;
    d_gm[1] = 0;
    for(auto & entry : obstacleBlocks) delete entry;
    obstacleBlocks.clear();
  }

 protected:

/*
  inline void rotate(Real p[2]) const
  {
    const Real x = p[0], y = p[1];
    p[0] =  x*std::cos(orientation) + y*std::sin(orientation);
    p[1] = -x*std::sin(orientation) + y*std::cos(orientation);
  }
*/
 public:
  Shape( SimulationData& s, ArgumentParser& p, double C[2] ) :
  sim(s), origC{C[0],C[1]}, origAng( p("-angle").asDouble(0)*M_PI/180 ),
  center{C[0],C[1]}, centerOfMass{C[0],C[1]}, orientation(origAng),
  rhoS( p("-rhoS").asDouble(1) ),
  bForced( p("-bForced").asBool(false) ),
  bFixed( p("-bFixed").asBool(false) ),
  bForcedx(p("-bForcedx").asBool(bForced)),
  bForcedy(p("-bForcedy").asBool(bForced)),
  bBlockang( p("-bBlockAng").asBool(bForcedx || bForcedy) ),
  bFixedx(p("-bFixedx" ).asBool(bFixed) ),
  bFixedy(p("-bFixedy" ).asBool(bFixed) ),
  forcedu( - p("-xvel").asDouble(0) ),
  forcedv( - p("-yvel").asDouble(0) )
  {  }

  virtual ~Shape() {
    for(auto & entry : obstacleBlocks) delete entry;
    obstacleBlocks.clear();
  }

  virtual Real getCharLength() const = 0;
  virtual void create(const vector<BlockInfo>& vInfo) = 0;

  virtual void updatePosition(double dt);

  void setCentroid(double C[2])
  {
    this->center[0] = C[0];
    this->center[1] = C[1];
    const double cost = std::cos(this->orientation);
    const double sint = std::sin(this->orientation);
    this->centerOfMass[0] = C[0] - cost*this->d_gm[0] + sint*this->d_gm[1];
    this->centerOfMass[1] = C[1] - sint*this->d_gm[0] - cost*this->d_gm[1];
  }

  void setCenterOfMass(double com[2])
  {
    this->centerOfMass[0] = com[0];
    this->centerOfMass[1] = com[1];
    const double cost = std::cos(this->orientation);
    const double sint = std::sin(this->orientation);
    this->center[0] = com[0] + cost*this->d_gm[0] - sint*this->d_gm[1];
    this->center[1] = com[1] + sint*this->d_gm[0] + cost*this->d_gm[1];
  }

  void getCentroid(double centroid[2]) const
  {
    centroid[0] = this->center[0];
    centroid[1] = this->center[1];
  }

  virtual void getCenterOfMass(double com[2]) const
  {
    com[0] = this->centerOfMass[0];
    com[1] = this->centerOfMass[1];
  }

  void getLabPosition(double com[2]) const
  {
    com[0] = this->labCenterOfMass[0];
    com[1] = this->labCenterOfMass[1];
  }

  double getU() const { return u; }
  double getV() const { return v; }
  double getW() const { return omega; }

  double getOrientation() const
  {
    return this->orientation;
  }
  void setOrientation(const double angle)
  {
    this->orientation = angle;
  }

  virtual Real getMinRhoS() const;
  virtual bool bVariableDensity() const;
  virtual void outputSettings(ostream &outStream) const;

  struct Integrals {
    const double x, y, m, j, u, v, a;
    Integrals(double _x, double _y, double _m, double _j, double _u, double _v, double _a) :
    x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
    Integrals(const Integrals&c) :
      x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
  };

  Integrals integrateObstBlock(const vector<BlockInfo>& vInfo);

  void removeMoments(const vector<BlockInfo>& vInfo);

  virtual void computeVelocities();

  void updateLabVelocity( int nSum[2], double uSum[2] );

  void penalize();

  void diagnostics();

  void characteristic_function();
  void deformation_velocities();

  template <typename Kernel>
  void compute(const Kernel& kernel, const vector<BlockInfo>& vInfo)
  {
    #pragma omp parallel
    {
      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(dynamic)
      for (size_t i=0; i<vInfo.size(); i++)
      {
        const auto pos = obstacleBlocks[vInfo[i].blockID];
        if(pos == nullptr) continue; //obstacle is not in the block
        mylab.load(vInfo[i], 0);
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(mylab, vInfo[i], b, pos);
      }
    }
  }

  template <typename Kernel>
  void compute_surface(const Kernel& kernel, const vector<BlockInfo>& vInfo)
  {
    #pragma omp parallel
    {
      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(dynamic)
      for (size_t i=0; i<vInfo.size(); i++)
      {
        const auto pos = obstacleBlocks[vInfo[i].blockID];
        if(pos == nullptr) continue; //obstacle is not in the block
        assert(pos->filled);
        if(!pos->n_surfPoints) continue; //does not contain surf points

        mylab.load(vInfo[i], 0);
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(mylab, vInfo[i], b, pos);
      }
    }
  }

  void computeForces();
};

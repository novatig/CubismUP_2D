//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "SimulationData.h"
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
  double orientation = origAng;

  const double rhoS;
  const bool bFixed;
  const bool bFixedx;
  const bool bFixedy;
  const bool bForced;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const double forcedu;
  const double forcedv;
  const double forcedomega;

  double M = 0;
  double J = 0;
  double u = forcedu; // in lab frame, not sim frame
  double v = forcedv; // in lab frame, not sim frame
  double omega = forcedomega;
  double fluidAngMom = 0;
  double fluidMomX = 0;
  double fluidMomY = 0;
  double penalDX = 0;
  double penalDY = 0;
  double penalM = 0;
  double penalJ = 0;
  double appliedForceX = 0;
  double appliedForceY = 0;
  double appliedTorque = 0;

  double perimeter=0, forcex=0, forcey=0, forcex_P=0, forcey_P=0;
  double forcex_V=0, forcey_V=0, torque=0, torque_P=0, torque_V=0;
  double drag=0, thrust=0, circulation=0, Pout=0, PoutBnd=0, defPower=0;
  double defPowerBnd=0, Pthrust=0, Pdrag=0, EffPDef=0, EffPDefBnd=0;

  virtual void resetAll()
  {
    center[0] = origC[0];
    center[1] = origC[1];
    centerOfMass[0] = origC[0];
    centerOfMass[1] = origC[1];
    labCenterOfMass[0] = 0;
    labCenterOfMass[1] = 0;
    orientation = origAng;
    M = 0;
    J = 0;
    u = forcedu;
    v = forcedv;
    omega = forcedomega;
    fluidMomX = 0;
    fluidMomY = 0;
    fluidAngMom = 0;
    appliedForceX = 0;
    appliedForceY = 0;
    appliedTorque = 0;
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
  Shape( SimulationData& s, cubism::ArgumentParser& p, double C[2] );

  virtual ~Shape();

  virtual Real getCharLength() const = 0;
  virtual Real getCharSpeed() const {
    return std::sqrt(forcedu*forcedu + forcedv*forcedv);
  }
  virtual Real getCharMass() const;
  virtual Real getMaxVel() const;

  virtual void create(const std::vector<cubism::BlockInfo>& vInfo) = 0;

  virtual void updateVelocity(double dt);
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
  virtual void outputSettings(std::ostream &outStream) const;

  struct Integrals {
    const double x, y, m, j, u, v, a;
    Integrals(double _x, double _y, double _m, double _j, double _u, double _v, double _a) :
    x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
    Integrals(const Integrals&c) :
      x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
  };

  Integrals integrateObstBlock(const std::vector<cubism::BlockInfo>& vInfo);

  virtual void removeMoments(const std::vector<cubism::BlockInfo>& vInfo);

  void updateLabVelocity( int mSum[2], double uSum[2] );

  void penalize();

  void diagnostics();

  virtual void computeForces();
};

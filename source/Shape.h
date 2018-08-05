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
  std::map<int,ObstacleBlock*> obstacleBlocks;
  // general quantities
  const Real origC[2], origAng;
  Real center[2]; // for single density, this corresponds to centerOfMass
  Real centerOfMass[2];
  Real d_gm[2]; // distance of center of geometry to center of mass
  Real labCenterOfMass[2] = {0,0};
  Real orientation;
  Real M = 0;
  Real V = 0;
  Real J = 0;
  Real u = 0; // in lab frame, not sim frame
  Real v = 0; // in lab frame, not sim frame
  Real omega = 0;
  Real computedu = 0;
  Real computedv = 0;
  Real computedo = 0;

  Real perimeter=0, forcex=0, forcey=0, forcex_P=0, forcey_P=0;
  Real forcex_V=0, forcey_V=0, torque=0, torque_P=0, torque_V=0;
  Real drag=0, thrust=0, circulation=0, Pout=0, PoutBnd=0, defPower=0;
  Real defPowerBnd=0, Pthrust=0, Pdrag=0, EffPDef=0, EffPDefBnd=0;

  const Real rhoS;
  const bool bForced;
  const bool bFixed;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const bool bFixedx;
  const bool bFixedy;
  const Real forcedu;
  const Real forcedv;

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
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();
  }

 protected:

  Real smoothHeaviside(Real rR, Real radius, Real eps) const
  {
    if (rR < radius-eps*.5) return (Real) 1.;
    else if (rR > radius+eps*.5) return (Real) 0.;
    else return (Real) ((1.+cos(M_PI*((rR-radius)/eps+.5)))*.5);
  }

  inline void rotate(Real p[2]) const
  {
    const Real x = p[0], y = p[1];
    p[0] =  x*std::cos(orientation) + y*std::sin(orientation);
    p[1] = -x*std::sin(orientation) + y*std::cos(orientation);
  }

 public:
  Shape( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  sim(s), origC{C[0],C[1]}, origAng( p("-angle").asDouble(0)*M_PI/180 ),
  center{C[0],C[1]}, centerOfMass{C[0],C[1]},
  d_gm{0,0}, orientation(origAng),
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
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();
  }

  virtual Real getCharLength() const = 0;
  virtual void create(const vector<BlockInfo>& vInfo) = 0;

  virtual void updatePosition(double dt);

  void setCentroid(Real centroid[2])
  {
    center[0] = centroid[0];
    center[1] = centroid[1];

    centerOfMass[0] = center[0] - std::cos(orientation)*d_gm[0] + std::sin(orientation)*d_gm[1];
    centerOfMass[1] = center[1] - std::sin(orientation)*d_gm[0] - std::cos(orientation)*d_gm[1];
  }

  void setCenterOfMass(Real com[2])
  {
    centerOfMass[0] = com[0];
    centerOfMass[1] = com[1];

    center[0] = centerOfMass[0] + std::cos(orientation)*d_gm[0] - std::sin(orientation)*d_gm[1];
    center[1] = centerOfMass[1] + std::sin(orientation)*d_gm[0] + std::cos(orientation)*d_gm[1];
  }

  void getCentroid(Real centroid[2]) const
  {
    centroid[0] = center[0];
    centroid[1] = center[1];
  }

  virtual void getCenterOfMass(Real com[2]) const
  {
    com[0] = centerOfMass[0];
    com[1] = centerOfMass[1];
  }

  void getLabPosition(Real com[2]) const
  {
    com[0] = labCenterOfMass[0];
    com[1] = labCenterOfMass[1];
  }

  Real getU() const { return u; }
  Real getV() const { return v; }
  Real getW() const { return omega; }

  Real getOrientation() const
  {
    return orientation;
  }
  void setOrientation(const Real angle)
  {
    orientation = angle;
  }

  virtual Real getMinRhoS() const;

  virtual void outputSettings(ostream &outStream) const;

  struct Integrals {
    const Real m, V, j, u, v, a, x, y, X, Y;

    Integrals(Real _m, Real _V, Real _j, Real _u, Real _v, Real _a,
      Real _x, Real _y, Real _X, Real _Y) :
    m(_m), V(_V), j(_j), u(_u), v(_v), a(_a), x(_x), y(_y), X(_X), Y(_Y) {}

    Integrals(const Integrals&c) :
    m(c.m),V(c.V),j(c.j),u(c.u),v(c.v),a(c.a),x(c.x),y(c.y),X(c.X),Y(c.Y){}
  };

  Integrals integrateObstBlock(const vector<BlockInfo>& vInfo);

  void removeMoments(const vector<BlockInfo>& vInfo);

  void computeVelocities();

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
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue;
        mylab.load(vInfo[i], 0);
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(mylab, vInfo[i], b, pos->second);
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
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue; //obstacle is not in the block
        assert(pos->second->filled);
        if(!pos->second->n_surfPoints) continue; //does not contain surf points

        mylab.load(vInfo[i], 0);
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(mylab, vInfo[i], b, pos->second);
      }
    }
  }

  void computeForces();
};

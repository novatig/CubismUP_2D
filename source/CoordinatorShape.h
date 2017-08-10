//
//  CoordinatorComputeShape.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorComputeShape_h
#define CubismUP_2D_CoordinatorComputeShape_h

#include "GenericCoordinator.h"
#include "Shape.h"

class CoordinatorComputeShape : public GenericCoordinator
{
 protected:
  const Real*const uBody;
  const Real*const vBody;
  const Real*const omegaBody;
  Shape*const shape;

 public:
  CoordinatorComputeShape(const Real*const u, const Real*const v, const Real*const w, Shape*const s, FluidGrid*const g) :
  GenericCoordinator(g), uBody(u), vBody(v), omegaBody(w), shape(s) { }

  void operator()(const double dt)
  {
    check("shape - start");
    const Real ub[2] = { *uBody, *vBody };
    shape->updatePosition(ub, *omegaBody, dt);
    const Real domainSize[2] = {
      grid->getBlocksPerDimension(0)*FluidBlock::sizeX*vInfo[0].h_gridpoint,
      grid->getBlocksPerDimension(1)*FluidBlock::sizeY*vInfo[0].h_gridpoint
    };
    Real p[2] = {0,0};
    shape->getCentroid(p);

    if (p[0]<0 || p[0]>domainSize[0] || p[1]<0 || p[1]>domainSize[1]) {
      cout << "Body out of domain: " << p[0] << " " << p[1] << endl;
      exit(0);
    }

    #pragma omp parallel for schedule(static)
    for(int i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy).rho = 1;
        b(ix,iy).tmp = 0;
      }
    }
    shape->create(vInfo);
    check("shape - end");
  }

  string getName()
  {
    return "ComputeShape";
  }
};

class CoordinatorBodyVelocities : public GenericCoordinator
{
protected:
  Real* const uBody;
  Real* const vBody;
  Real* const omegaBody;
  Real* const time;
  const Real* const lambda;
  Shape* const shape;

public:
  CoordinatorBodyVelocities(Real*const u, Real*const v, Real*const w, Shape*const s, Real*const l, Real*const t, FluidGrid*const g) : GenericCoordinator(g), uBody(u), vBody(v), omegaBody(w), lambda(l), shape(s), time(t) { }

  void operator()(const double dt)
  {
    shape->computeVelocities(uBody, vBody, omegaBody, time, dt, vInfo);
    //act is here as it allows modifying velocities before penalization
    shape->act(uBody, vBody, omegaBody, dt);
  }

  string getName()
  {
    return "BodyVelocities";
  }
};

class CoordinatorPenalization : public GenericCoordinator
{
 protected:
  const Real* const uBody;
  const Real* const vBody;
  const Real* const omegaBody;
  const Real* const lambda;
  Shape* const shape;

 public:
  CoordinatorPenalization(Real*uBody, Real*vBody, Real*omegaBody, Shape*shape, Real*lambda, FluidGrid*grid) :
    GenericCoordinator(grid), uBody(uBody), vBody(vBody), omegaBody(omegaBody), shape(shape), lambda(lambda) { }

  void operator()(const double dt)
  {
    check("penalization - start");
    shape->penalize(*uBody, *vBody, *omegaBody, dt, *lambda, vInfo);
    check("penalization - end");
  }

  string getName()
  {
    return "Penalization";
  }
};

class CoordinatorComputeForces : public GenericCoordinator
{
 protected:
  const Real* const uBody;
  const Real* const vBody;
  const Real* const omegaBody;
  const Real* const time;
  const Real* const NU;
  const  int* const stepID;
  const bool* const bDump;
  Shape* const shape;

 public:
  CoordinatorComputeForces(const Real*const uBody, const Real*const vBody, const Real*const omegaBody, Shape*const shape, const Real*const time, const Real*const NU, const int*const stepID, const bool*const dump, FluidGrid*grid) : GenericCoordinator(grid), uBody(uBody), vBody(vBody), omegaBody(omegaBody), shape(shape), time(time), NU(NU), stepID(stepID), bDump(dump) { }

  void operator()(const double dt)
  {
    check("forces - start");
    shape->computeForces(*stepID, *time, dt, *NU, *uBody, *vBody, *omegaBody, *bDump, vInfo);
    check("forces - end");
  }

  string getName()
  {
    return "Forces";
  }
};
#endif

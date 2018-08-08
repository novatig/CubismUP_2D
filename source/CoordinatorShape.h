//
//  CoordinatorComputeShape.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "GenericCoordinator.h"
#include "Shape.h"

class CoordinatorComputeShape : public GenericCoordinator
{
 protected:
  const Real domainSize[2] = {
    (Real) sim.getH() * sim.grid->getBlocksPerDimension(0) * FluidBlock::sizeX,
    (Real) sim.getH() * sim.grid->getBlocksPerDimension(1) * FluidBlock::sizeY
  };

 public:
  CoordinatorComputeShape(SimulationData&s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("shape - start");

    for( const auto& shape : sim.shapes )
    {
      shape->updatePosition(dt);
      Real p[2] = {0,0};
      shape->getCentroid(p);
      if (p[0]<0 || p[0]>domainSize[0] || p[1]<0 || p[1]>domainSize[1]) {
        cout << "Body out of domain: " << p[0] << " " << p[1] << endl;
        exit(0);
      }
    }

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy).invRho = 1;
        b(ix,iy).tmp = 0;
      }
    }

    for( const auto& shape : sim.shapes ) shape->create(vInfo);

    check("shape - end");
  }

  string getName()
  {
    return "ComputeShape";
  }
};

class CoordinatorVelocities : public GenericCoordinator
{
 public:
  CoordinatorVelocities(SimulationData&s): GenericCoordinator(s) {}

  void operator()(const double dt)
  {
    int nSum[2] = {0, 0};
    double uSum[2] = {0, 0};
    for( const auto& shape : sim.shapes ) {
      shape->computeVelocities();
      shape->updateLabVelocity(nSum, uSum);
    }
    if(nSum[0]) sim.uinfx = uSum[0]/nSum[0];
    if(nSum[1]) sim.uinfy = uSum[1]/nSum[1];
  }

  string getName()
  {
    return "BodyVelocities";
  }
};

class CoordinatorPenalization : public GenericCoordinator
{
 public:
  CoordinatorPenalization(SimulationData&s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("penalization - start");
    for( const auto& shape : sim.shapes ) shape->penalize();
    check("penalization - end");
  }

  string getName()
  {
    return "Penalization";
  }
};

class CoordinatorComputeForces : public GenericCoordinator
{
 public:
  CoordinatorComputeForces(SimulationData&s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("forces - start");
    for( const auto& shape : sim.shapes ) {
      shape->diagnostics();
      shape->computeForces();
    }
    check("forces - end");
  }

  string getName()
  {
    return "Forces";
  }
};

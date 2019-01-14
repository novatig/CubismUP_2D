//
//  CoordinatorIC.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "GenericCoordinator.h"
#include "Shape.h"

class OperatorIC : public GenericOperator
{
  const SimulationData& sim;
 public:
  OperatorIC(const double dt, const SimulationData& s) : sim(s) {}
  ~OperatorIC() {}

  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    for( const auto& shape : sim.shapes ) shape->penalize();
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      block(ix,iy).u = 0;
      block(ix,iy).v = 0;
      block(ix,iy).invRho = 1;
      block(ix,iy).p = 0;
      block(ix,iy).pOld = 0;
      block(ix,iy).tmpU = 0;
      block(ix,iy).tmpV = 0;
      block(ix,iy).tmp  = 0;
    }
    for( const auto& shape : sim.shapes )
    {
      const std::vector<ObstacleBlock*>& obstacleBlocks = shape->obstacleBlocks;
      const Real u = shape->u, v = shape->v, omega = shape->omega;
      const double* const centerOfMass = shape->centerOfMass;
      const auto pos = obstacleBlocks[info.blockID];
      if(pos == nullptr) continue;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real chi = pos->chi[iy][ix];
        if (chi <= 0) continue;
        double p[2]; info.pos(p, ix, iy);
        p[0] -= centerOfMass[0]; p[1] -= centerOfMass[1];
        block(ix,iy).u = chi * (u -omega*p[1]);
        block(ix,iy).v = chi * (v +omega*p[0]);
      }
    }
  }
};

class CoordinatorIC : public GenericCoordinator
{
  public:
  CoordinatorIC(SimulationData& s) : GenericCoordinator(s) { }

  void operator()(const double dt) {
    #pragma omp parallel
    {
      OperatorIC kernel(dt, sim);
      #pragma omp for schedule(static)
      for (size_t i=0; i<vInfo.size(); i++)
        kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
    }
    check("IC - end");
  }

  string getName()
  {
    return "IC";
  }
};

class OperatorFadeOut : public GenericOperator
{
 private:
  const Real extent[2], fac;
  const int buffer;
  static constexpr int Z = 0;

  inline bool _is_touching(const BlockInfo& i, const Real h) const {
    Real min_pos[2], max_pos[2];
    i.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    i.pos(min_pos, 0, 0);
    const bool touchN = (Z+buffer)*h >= extent[1] - max_pos[1];
    const bool touchE = (Z+buffer)*h >= extent[0] - max_pos[0];
    const bool touchS = (Z+buffer)*h >= min_pos[1];
    const bool touchW = (Z+buffer)*h >= min_pos[0];
    return touchN || touchE || touchS || touchW;
  }

 public:
  OperatorFadeOut(int B, const Real E[2], const Real F): extent{E[0], E[1]},
  fac(F), buffer(B) {}

  void operator()(const BlockInfo& i, FluidBlock& b) const {
    const Real h = i.h_gridpoint, iWidth = 1/(buffer*h);
    if(_is_touching(i,h))
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      Real p[2];
      i.pos(p, ix, iy);
      const Real arg1= std::max((Real)0, (Z+buffer)*h -(extent[0]-p[0]) );
      const Real arg2= std::max((Real)0, (Z+buffer)*h -(extent[1]-p[1]) );
      const Real arg3= std::max((Real)0, (Z+buffer)*h -p[0] );
      const Real arg4= std::max((Real)0, (Z+buffer)*h -p[1] );
      const Real dist= std::max(std::max(arg1, arg2), std::max(arg3, arg4));
      const Real fade= std::max(1-fac, 1 - fac*(dist*iWidth)*(dist*iWidth));
      b(ix,iy).u = b(ix,iy).u*fade;
      b(ix,iy).v = b(ix,iy).v*fade;
    }
  }
};

class CoordinatorFadeOut : public GenericCoordinator
{
 protected:
  const int buffer;
  const Real ext[2] = {
    (Real) sim.getH() * sim.grid->getBlocksPerDimension(0) * FluidBlock::sizeX,
    (Real) sim.getH() * sim.grid->getBlocksPerDimension(1) * FluidBlock::sizeY
  };

 public:
  CoordinatorFadeOut(SimulationData& s, int _buf=8) : GenericCoordinator(s), buffer(_buf) { }

  void operator()(const double dt) {
    check((string)"FadeOut - start");
    #pragma omp parallel
    {
      OperatorFadeOut kernel(buffer, ext, sim.CFL);
      #pragma omp for schedule(dynamic)
      for(size_t i=0; i<vInfo.size(); i++)
        kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
    }
    check((string)"FadeOut - end");
  }

  string getName()
  {
    return "FadeOut";
  }
};

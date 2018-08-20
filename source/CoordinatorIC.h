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
 public:
  OperatorIC(const double dt) {}
  ~OperatorIC() {}

  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        Real p[2];
        info.pos(p, ix, iy);

        block(ix,iy).u = 0;
        block(ix,iy).v = 0;
        block(ix,iy).invRho = 1;

        block(ix,iy).p = 0;
        block(ix,iy).pOld = 0;

        block(ix,iy).tmpU = 0;
        block(ix,iy).tmpV = 0;
        block(ix,iy).tmp  = 0;
      }
  }
};

class CoordinatorIC : public GenericCoordinator
{
  public:
  CoordinatorIC(SimulationData& s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    #pragma omp parallel
    {
      OperatorIC kernel(dt);
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
  const Real extent[2];
  const int buffer;

  inline bool _is_touching(const BlockInfo& i, const Real h) const
  {
    Real min_pos[2], max_pos[2];
    i.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    i.pos(min_pos, 0, 0);
    const bool touchN = (1+buffer)*h >= extent[1] - max_pos[1];
    const bool touchE = (1+buffer)*h >= extent[0] - max_pos[0];
    const bool touchS = (1+buffer)*h >= min_pos[1];
    const bool touchW = (1+buffer)*h >= min_pos[0];
    return touchN || touchE || touchS || touchW;
  }

 public:
  OperatorFadeOut(int buf, const Real ext[2]): extent{ext[0],ext[1]}, buffer(buf) {}

  void operator()(const BlockInfo& i, FluidBlock& b) const {
    const Real h = i.h_gridpoint, iWidth = 1/(buffer*h), fac = 0.5*M_PI;
    //const int ax = info[0], dir = info[1];
    if(_is_touching(i,h))
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      Real p[2];
      i.pos(p, ix, iy);
      const Real arg1= std::max((Real)0, (1+buffer)*h -(extent[0]-p[0]) );
      const Real arg2= std::max((Real)0, (1+buffer)*h -(extent[1]-p[1]) );
      const Real arg3= std::max((Real)0, (1+buffer)*h -p[0] );
      const Real arg4= std::max((Real)0, (1+buffer)*h -p[1] );
      const Real dist= std::max(std::max(arg1, arg2), std::max(arg3, arg4));
      const Real fade= std::max((Real)0, std::cos(fac*dist*iWidth) );
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
  CoordinatorFadeOut(SimulationData& s, int _buf=4) : GenericCoordinator(s), buffer(_buf) { }

  void operator()(const double dt)
  {
    check((string)"FadeOut - start");
    //const int movey = fabs(uinfy-*vBody) > fabs(uinfx-*uBody);
    //const int dirs[2] = {*uBody-uinfx>0 ? 1 : -1, *vBody-uinfy>0 ? 1 : -1};
    //const int info[2] = {movey, dirs[movey]};

    #pragma omp parallel
    {
      OperatorFadeOut kernel(buffer, ext);
      #pragma omp for schedule(static)
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

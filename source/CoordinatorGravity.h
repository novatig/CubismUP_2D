//
//  CoordinatorGravity.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorGravity_h
#define CubismUP_2D_CoordinatorGravity_h

#include "GenericCoordinator.h"
#include "GenericOperator.h"

class CoordinatorGravity : public GenericCoordinator
{
 protected:
  const Real gravity[2] = { sim.gravity[0], sim.gravity[1] };

 public:
  CoordinatorGravity(SimulationData& s) : GenericCoordinator(s) { }

  inline void addHydrostaticPressure(const Real dt)
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const BlockInfo& info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy).u += dt*gravity[0]*(1 - b(ix,iy).invRho);
        b(ix,iy).v += dt*gravity[1]*(1 - b(ix,iy).invRho);
      }
    }
  }

  void operator()(const double dt)
  {
    check("gravity - start");
    addHydrostaticPressure(dt);
    /*
    BlockInfo * ary = &vInfo.front();
    const int N = vInfo.size();
    #pragma omp parallel
    {
      OperatorGravity kernel(gravity, dt);
      #pragma omp for schedule(static)
      for (int i=0; i<N; i++)
        kernel(ary[i], *(FluidBlock*)ary[i].ptrBlock);
    }
    */
    check("gravity - end");
  }

  string getName()
  {
    return "Gravity";
  }
};


#endif

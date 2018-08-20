//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//


#pragma once
#include "GenericCoordinator.h"

class OperatorDiffusion : public GenericLabOperator
{
 private:
  const double nu, dt;

 public:
  OperatorDiffusion(double _dt, double _nu) : nu(_nu), dt(_dt)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 2, 0,1);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }

  ~OperatorDiffusion() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real fac = nu * dt / (info.h_gridpoint*info.h_gridpoint);

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const FluidElement& phi  = lab(ix,iy);
      const FluidElement& phiN = lab(ix,iy+1);
      const FluidElement& phiS = lab(ix,iy-1);
      const FluidElement& phiE = lab(ix+1,iy);
      const FluidElement& phiW = lab(ix-1,iy);

      o(ix,iy).tmpU = phi.u + fac * (phiN.u +phiS.u +phiE.u +phiW.u -phi.u*4);
      o(ix,iy).tmpV = phi.v + fac * (phiN.v +phiS.v +phiE.v +phiW.v -phi.v*4);
    }
   }
};

template <typename Lab>
class CoordinatorDiffusion : public GenericCoordinator
{
  inline void reset()
  {
    #ifndef NDEBUG
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      BlockInfo info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          b(ix,iy).tmpU = 0;
          b(ix,iy).tmpV = 0;
        }
    }
    #endif
  };

  inline void update()
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      BlockInfo info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          b(ix,iy).u = b(ix,iy).tmpU;
          b(ix,iy).v = b(ix,iy).tmpV;
        }
    }
   }

  inline void diffuse(const double dt, const int stage)
  {
    #pragma omp parallel
    {
      OperatorDiffusion kernel(dt, sim.nu);

      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(static)
      for (size_t i=0; i<vInfo.size(); i++) {
        const BlockInfo& info = vInfo[i];
        mylab.load(info, 0);
        kernel(mylab, info, *(FluidBlock*)info.ptrBlock);
      }
    }
  }

 public:
  CoordinatorDiffusion(SimulationData& s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("diffusion - start");

    reset();
    diffuse(dt,0);
    update();

    check("diffusion - end");
  }

  string getName()
  {
    return "Diffusion";
  }
};

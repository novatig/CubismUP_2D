//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorDiffusion_h
#define CubismUP_2D_CoordinatorDiffusion_h

#include "GenericCoordinator.h"

class OperatorViscousDrag : public GenericLabOperator
{
 private:
  const double dt;
  Real viscousDrag = 0;

 public:
  OperatorViscousDrag(double _dt) : dt(_dt)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 1, 5);
    stencil_start[0] = -2; stencil_start[1] = -2; stencil_start[2] = 0;
    stencil_end[0] = 3; stencil_end[1] = 3; stencil_end[2] = 1;
  }

  ~OperatorViscousDrag() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
  {
    const Real prefactor = 1. / (info.h_gridpoint*info.h_gridpoint);
    viscousDrag = 0;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const Real phi  = lab(ix,iy).tmp;
      const Real phiN = lab(ix,iy+1).tmp;
      const Real phiS = lab(ix,iy-1).tmp;
      const Real phiE = lab(ix+1,iy).tmp;
      const Real phiW = lab(ix-1,iy).tmp;
      viscousDrag += prefactor * (phiN + phiS + phiE + phiW - 4.*phi);
    }
  }

  inline Real getDrag()
  {
    return viscousDrag;
  }
};

class OperatorDiffusion : public GenericLabOperator
{
 private:
  const double mu, dt;

 public:
  OperatorDiffusion(double _dt, double _mu) : mu(_mu), dt(_dt)
  {
    #ifndef _MULTIPHASE_
      stencil = StencilInfo(-1,-1,0, 2,2,1, false, 2, 0,1);
    #else
      stencil = StencilInfo(-1,-1,0, 2,2,1, false, 3, 0,1,4);
    #endif
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }

  ~OperatorDiffusion() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real prefactor = mu * dt / (info.h_gridpoint*info.h_gridpoint);

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const FluidElement& phi  = lab(ix,iy);
      const FluidElement& phiN = lab(ix,iy+1);
      const FluidElement& phiS = lab(ix,iy-1);
      const FluidElement& phiE = lab(ix+1,iy);
      const FluidElement& phiW = lab(ix-1,iy);
      const Real fac = prefactor/phi.rho;
      o(ix,iy).tmpU = phi.u + fac * (phiN.u +phiS.u +phiE.u +phiW.u -phi.u*4);
      o(ix,iy).tmpV = phi.v + fac * (phiN.v +phiS.v +phiE.v +phiW.v -phi.v*4);
      #ifdef _MULTIPHASE_
      o(ix,iy).tmp=phi.rho+fac*(phiN.rho+phiS.rho+phiE.rho+phiW.rho-phi.rho*4);
      #endif
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
    for(size_t i=0; i<vInfo.size(); i++)
    {
      BlockInfo info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
        {
          b(ix,iy).tmpU = 0;
          b(ix,iy).tmpV = 0;
          #ifdef _MULTIPHASE_
            b(ix,iy).tmp = 0;
          #endif
        }
    }
    #endif
  };

  inline void update()
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      BlockInfo info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
        {
          b(ix,iy).u = b(ix,iy).tmpU;
          b(ix,iy).v = b(ix,iy).tmpV;
          #ifdef _MULTIPHASE_
            b(ix,iy).rho = b(ix,iy).tmp;
          #endif
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

  inline void drag()
  {
    /*
    BlockInfo * ary = &vInfo.front();
    #pragma omp parallel
    {
      OperatorViscousDrag kernel(0);
      Real tmpDrag = 0;

      Lab mylab;
      mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(static) reduction(+:tmpDrag)
      for (size_t i=0; i<vInfo.size(); i++) {
        mylab.load(ary[i], 0);
        kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
        tmpDrag += kernel.getDrag();
      }
    }
    shape->dragV = tmpDrag*coeff;
    */
  }

 public:
  CoordinatorDiffusion(SimulationData& s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("diffusion - start");

    reset();
    diffuse(dt,0);
    update();
    //drag();

    check("diffusion - end");
  }

  string getName()
  {
    return "Diffusion";
  }
};

#endif

//
//  CoordinatorAdvection.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorAdvection_h
#define CubismUP_2D_CoordinatorAdvection_h

#include "GenericCoordinator.h"
#include <cmath>

class OperatorAdvectionFD : public GenericLabOperator
{
 private:
  const double dt;
  const Real uinf;
  const Real vinf;

 public:
  OperatorAdvectionFD(double _dt, const Real u, const Real v)
  : dt(_dt), uinf(u), vinf(v)
  {
    #ifndef _MULTIPHASE_
      stencil = StencilInfo(-1,-1,0, 2,2,1, false, 2, 0,1);
    #else
      stencil = StencilInfo(-1,-1,0, 2,2,1, false, 3, 0,1,4);
    #endif
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorAdvectionFD() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const double dh = info.h_gridpoint, fac = -.5*dt/info.h_gridpoint;

    for (int iy=0; iy<FluidBlock::sizeY; ++iy)
    for (int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const Real u = lab(ix,iy).u + uinf;
      const Real v = lab(ix,iy).v + vinf;

      const Real dudx = lab(ix+1,iy).u - lab(ix-1,iy).u;
      const Real dudy = lab(ix,iy+1).u - lab(ix,iy-1).u;
      const Real dvdx = lab(ix+1,iy).v - lab(ix-1,iy).v;
      const Real dvdy = lab(ix,iy+1).v - lab(ix,iy-1).v;

      o(ix,iy).tmpU = lab(ix,iy).u + fac*(u * dudx + v * dudy);
      o(ix,iy).tmpV = lab(ix,iy).v + fac*(u * dvdx + v * dvdy);

      #ifdef _MULTIPHASE_
        const Real drdx = lab(ix+1,iy).rho - lab(ix-1,iy).rho;
        const Real drdy = lab(ix,iy+1).rho - lab(ix,iy-1).rho;

        o(ix,iy).tmp  = lab(ix,iy).rho + fac*(u * drdx + v * drdy);
      #endif
    }
  }
};

class OperatorAdvectionUpwind3rdOrder : public GenericLabOperator
{
 private:
  const double dt;
  const Real uinf;
  const Real vinf;

 public:
  OperatorAdvectionUpwind3rdOrder(double _dt, const Real u, const Real v)
  : dt(_dt), uinf(u), vinf(v)
    {
      #ifndef _MULTIPHASE_
        stencil = StencilInfo(-2,-2,0, 3,3,1, false, 2, 0,1);
      #else
        stencil = StencilInfo(-2,-2,0, 3,3,1, false, 3, 0,1,4);
      #endif
      stencil_start[0] = -2; stencil_start[1] = -2; stencil_start[2] = 0;
      stencil_end[0] = 3; stencil_end[1] = 3; stencil_end[2] = 1;
    }

  ~OperatorAdvectionUpwind3rdOrder() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real factor = -dt/(6.*info.h_gridpoint);

    for (int iy=0; iy<FluidBlock::sizeY; ++iy)
      for (int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        const Real uPx = lab(ix+2,iy  ).u, vPx = lab(ix+2,iy  ).v;
        const Real upx = lab(ix+1,iy  ).u, vpx = lab(ix+1,iy  ).v;
        const Real uPy = lab(ix  ,iy+2).u, vPy = lab(ix  ,iy+2).v;
        const Real upy = lab(ix  ,iy+1).u, vpy = lab(ix  ,iy+1).v;
        const Real ucc = lab(ix  ,iy  ).u, vcc = lab(ix  ,iy  ).v;
        const Real ulx = lab(ix-1,iy  ).u, vlx = lab(ix-1,iy  ).v;
        const Real uLx = lab(ix-2,iy  ).u, vLx = lab(ix-2,iy  ).v;
        const Real uly = lab(ix  ,iy-1).u, vly = lab(ix  ,iy-1).v;
        const Real uLy = lab(ix  ,iy-2).u, vLy = lab(ix  ,iy-2).v;

        const Real u = ucc + uinf, v = vcc + vinf;
        const Real dux = u>0 ? 2*upx+3*ucc-6*ulx+uLx : -uPx+6*upx-3*ucc-2*ulx;
        const Real duy = v>0 ? 2*upy+3*ucc-6*uly+uLy : -uPy+6*upy-3*ucc-2*uly;
        const Real dvx = u>0 ? 2*vpx+3*vcc-6*vlx+vLx : -vPx+6*vpx-3*vcc-2*vlx;
        const Real dvy = v>0 ? 2*vpy+3*vcc-6*vly+vLy : -vPy+6*vpy-3*vcc-2*vly;

        o(ix,iy).tmpU = ucc + factor*(u*dux + v*duy);
        o(ix,iy).tmpV = vcc + factor*(u*dvx + v*dvy);

        #ifdef _MULTIPHASE_
          const Real drdx[2] = {  2*lab(ix+1,iy  ).rho + 3*lab(ix  ,iy  ).rho - 6*lab(ix-1,iy  ).rho +   lab(ix-2,iy  ).rho,
                       -  lab(ix+2,iy  ).rho + 6*lab(ix+1,iy  ).rho - 3*lab(ix  ,iy  ).rho - 2*lab(ix-1,iy  ).rho};

          const Real drdy[2] = {  2*lab(ix  ,iy+1).rho + 3*lab(ix  ,iy  ).rho - 6*lab(ix  ,iy-1).rho +   lab(ix  ,iy-2).rho,
                       -  lab(ix  ,iy+2).rho + 6*lab(ix  ,iy+1).rho - 3*lab(ix  ,iy  ).rho - 2*lab(ix  ,iy-1).rho};


          o(ix,iy).tmp  = o(ix,iy).rho + factor*(max(u,(Real)0) * drdx[0] + min(u,(Real)0) * drdx[1] +
                         max(v,(Real)0) * drdy[0] + min(v,(Real)0) * drdy[1]);
        #endif // _MULTIPHASE_
    }
  }
};

template <typename Lab>
class CoordinatorAdvection : public GenericCoordinator
{
 public:
    CoordinatorAdvection(SimulationData& s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("advection - start");
    #ifndef NDEBUG
      #pragma omp parallel for schedule(static)
      for(size_t i=0; i<vInfo.size(); i++) {
        BlockInfo info = vInfo[i];
        FluidBlock& b = *(FluidBlock*)info.ptrBlock;

        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
          for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            b(ix,iy).tmpU = 0;
            b(ix,iy).tmpV = 0;
            #ifdef _MULTIPHASE_
              b(ix,iy).tmp = 0;
            #endif // _MULTIPHASE_
          }
      }
    #endif
    ////////////////////////////////////////////////////////////////////////////


    #pragma omp parallel
    {
      OperatorAdvectionUpwind3rdOrder kernel(dt, sim.uinfx, sim.uinfy);
      //OperatorAdvectionFD kernel(dt,uBody,vBody);

      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(static)
      for (size_t i=0; i<vInfo.size(); i++) {
        const BlockInfo& info = vInfo[i];
        mylab.load(info, 0);
        kernel(mylab, info, *(FluidBlock*)info.ptrBlock);
      }
    }

    ///////////////////////////////////////////////////////////////////////////


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
          #ifdef _MULTIPHASE_ // threshold density
          b(ix,iy).rho = b(ix,iy).tmp;
          #endif // _MULTIPHASE_
        }
    }

    check("advection - end");
  }

  string getName()
  {
    return "Advection";
  }
};

#endif

//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once
#define UPWIND
#include "GenericCoordinator.h"

class OperatorMultistep : public GenericLabOperator
{
 private:
  const double mu, dt;
  const Real g[2];
  const Real uinf;
  const Real vinf;

 public:
  OperatorMultistep(double _dt, double _mu, Real _g[2], Real u, Real v) :
  mu(_mu), dt(_dt), g{_g[0],_g[1]}, uinf(u), vinf(v)
  {
    #ifdef UPWIND
      stencil = StencilInfo(-2,-2,0, 3,3,1, false, 2, 0,1);
      stencil_start[0] = -2; stencil_start[1] = -2; stencil_start[2] = 0;
      stencil_end[0] = 3; stencil_end[1] = 3; stencil_end[2] = 1;
    #else
      stencil = StencilInfo(-1,-1,0, 2,2,1, false, 2, 0,1);
      stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
      stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
    #endif
  }

  ~OperatorMultistep() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
  {
    const Real diffac = (mu/info.h_gridpoint) * (dt/info.h_gridpoint);
    #ifdef UPWIND
      const Real advfac = -dt/(6*info.h_gridpoint);
    #else
      const Real advfac = -dt/(2*info.h_gridpoint);
    #endif

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const Real upx = lab(ix+1,iy  ).u, vpx = lab(ix+1,iy  ).v;
      const Real upy = lab(ix  ,iy+1).u, vpy = lab(ix  ,iy+1).v;
      const Real ucc = lab(ix  ,iy  ).u, vcc = lab(ix  ,iy  ).v;
      const Real ulx = lab(ix-1,iy  ).u, vlx = lab(ix-1,iy  ).v;
      const Real uly = lab(ix  ,iy-1).u, vly = lab(ix  ,iy-1).v;
      const Real u = ucc + uinf, v = vcc + vinf;

      #ifdef UPWIND
        const Real uPx = lab(ix+2,iy  ).u, vPx = lab(ix+2,iy  ).v;
        const Real uPy = lab(ix  ,iy+2).u, vPy = lab(ix  ,iy+2).v;
        const Real uLx = lab(ix-2,iy  ).u, vLx = lab(ix-2,iy  ).v;
        const Real uLy = lab(ix  ,iy-2).u, vLy = lab(ix  ,iy-2).v;
        const Real dux = u>0 ? 2*upx+3*ucc-6*ulx+uLx : -uPx+6*upx-3*ucc-2*ulx;
        const Real duy = v>0 ? 2*upy+3*ucc-6*uly+uLy : -uPy+6*upy-3*ucc-2*uly;
        const Real dvx = u>0 ? 2*vpx+3*vcc-6*vlx+vLx : -vPx+6*vpx-3*vcc-2*vlx;
        const Real dvy = v>0 ? 2*vpy+3*vcc-6*vly+vLy : -vPy+6*vpy-3*vcc-2*vly;
      #else
        const Real dux = lab(ix+1,iy).u - lab(ix-1,iy).u;
        const Real duy = lab(ix,iy+1).u - lab(ix,iy-1).u;
        const Real dvx = lab(ix+1,iy).v - lab(ix-1,iy).v;
        const Real dvy = lab(ix,iy+1).v - lab(ix,iy-1).v;
      #endif

      const Real gravFac = dt * (1 - o(ix,iy).invRho);
      const Real duGrav = g[0] * gravFac, dvGrav = g[1] * gravFac;
      const Real duDiff = diffac * (upx + upy + ulx + uly - 4 * ucc);
      const Real dvDiff = diffac * (vpx + vpy + vlx + vly - 4 * vcc);
      const Real duAdvc = advfac * (u*dux + v*duy);
      const Real dvAdvc = advfac * (u*dvx + v*dvy);

      o(ix,iy).tmpU = ucc + duGrav + duDiff + duAdvc;
      o(ix,iy).tmpV = vcc + dvGrav + dvDiff + dvAdvc;
    }
  }
};

template <typename Lab>
class CoordinatorMultistep : public GenericCoordinator
{
  inline void update()
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      const BlockInfo& info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy).u = b(ix,iy).tmpU;
        b(ix,iy).v = b(ix,iy).tmpV;
      }
    }
  }
  void removeIncomingMom()
  {
    int count = 0;
    Real u = 0, v = 0;
    for(size_t i=0; i<vInfo.size(); i++) {
      if(vInfo[i].origin[0] > 1e-10) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      count += FluidBlock::sizeY;
      for(int iy=0;iy<FluidBlock::sizeY;++iy){ u+=b(0,iy).u; v+=b(0,iy).v; }
    }
    //cout << "BC:" << u << " "  << v << " " << count << endl;
    const Real UBC = u/count, VBC = v/count;
    for(size_t i=0; i<vInfo.size(); i++) {
      if(vInfo[i].origin[0] > 1e-10) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      for(int iy=0;iy<FluidBlock::sizeY;++iy){ b(0,iy).u-=UBC; b(0,iy).v-=VBC; }
    }
  }

 public:
  CoordinatorMultistep(SimulationData& s) : GenericCoordinator(s) { }

  void operator()(const double dt)
  {
    check("multistep - start");
    removeIncomingMom();
    #pragma omp parallel
    {
      OperatorMultistep kernel(dt, sim.nu, sim.gravity, sim.uinfx, sim.uinfy);

      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start,kernel.stencil_end,false);

      #pragma omp for schedule(static)
      for (size_t i=0; i<vInfo.size(); i++) {
        const BlockInfo& info = vInfo[i];
        mylab.load(info, 0);
        kernel(mylab, info, *(FluidBlock*)info.ptrBlock);
      }
    }
    update();

    check("multistep - end");
  }

  string getName()
  {
    return "Multistep";
  }
};

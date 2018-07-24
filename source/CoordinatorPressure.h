//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorPressure_h
#define CubismUP_2D_CoordinatorPressure_h

#include "GenericCoordinator.h"
#ifdef FREESPACE
  #include "PoissonSolverScalarFFTW_freespace.h"
#else
  #include "PoissonSolverScalarFFTW_periodic.h"
#endif

//#define _HYDROSTATIC_
class OperatorDivergenceSplit : public GenericLabOperator
{
 private:
  const double dt;
  const Real rho0;

  static inline Real mean(const Real a, const Real b) {return .5 * (a + b);}
  //harmonic mean: (why?)
  //inline Real mean(const Real a, const Real b) const {return 2.*a*b/(a+b);}

 public:
  OperatorDivergenceSplit(double _dt, double _rho0) : dt(_dt), rho0(_rho0)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 5, 0,1,4,6,7);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }

  ~OperatorDivergenceSplit() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real invH2 = 1./(info.h_gridpoint*info.h_gridpoint);
    const Real factor = rho0 * 0.5/(info.h_gridpoint * dt);

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      FluidElement& phi  = lab(ix  ,iy  );
      FluidElement& phiN = lab(ix  ,iy+1);
      FluidElement& phiS = lab(ix  ,iy-1);
      FluidElement& phiE = lab(ix+1,iy  );
      FluidElement& phiW = lab(ix-1,iy  );

      const Real pN = 2*phiN.p - phiN.pOld;
      const Real pS = 2*phiS.p - phiS.pOld;
      const Real pW = 2*phiW.p - phiW.pOld;
      const Real pE = 2*phiE.p - phiE.pOld;
      const Real p  = 2*phi.p  - phi.pOld;

      const Real fN = (1-rho0/mean(phiN.rho,phi.rho))*(pN - p ); // x1/h later
      const Real fS = (1-rho0/mean(phiS.rho,phi.rho))*(p  - pS);
      const Real fE = (1-rho0/mean(phiE.rho,phi.rho))*(pE - p );
      const Real fW = (1-rho0/mean(phiW.rho,phi.rho))*(p  - pW);

      const Real divUfac = factor * (phiE.u-phiW.u + phiN.v-phiS.v);
      const Real hatPfac =  invH2 * (fE - fW + fN - fS);

      o(ix, iy).tmp  = divUfac + hatPfac;
    }
  }
};

class OperatorGradPSplit : public GenericLabOperator
{
 private:
  const Real rho0;
  const double dt;

 public:
  OperatorGradPSplit(double _dt, double _rho0) : rho0(_rho0), dt(_dt)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 3, 5,6,7);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }

  ~OperatorGradPSplit() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const double dh = info.h_gridpoint;
    const Real prefactor = -.5 * dt / dh;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const FluidElement& phi  = lab(ix  ,iy  );
      const FluidElement& phiN = lab(ix  ,iy+1);
      const FluidElement& phiS = lab(ix  ,iy-1);
      const FluidElement& phiE = lab(ix+1,iy  );
      const FluidElement& phiW = lab(ix-1,iy  );

      const Real pN = 2*phiN.p - phiN.pOld;
      const Real pS = 2*phiS.p - phiS.pOld;
      const Real pW = 2*phiW.p - phiW.pOld;
      const Real pE = 2*phiE.p - phiE.pOld;

      // divU contains the pressure correction after the Poisson solver
      o(ix,iy).u += prefactor/rho0 * (phiE.tmp - phiW.tmp);
      o(ix,iy).v += prefactor/rho0 * (phiN.tmp - phiS.tmp);

      // add the split explicit term
      o(ix,iy).u += prefactor * (pE - pW) * (1./phi.rho - 1/rho0);
      o(ix,iy).v += prefactor * (pN - pS) * (1./phi.rho - 1/rho0);
    }
  }
};

class OperatorPressureDrag : public GenericLabOperator
{
 private:
  const double dt;
  Real pressureDrag[2] = {0, 0};

 public:
  OperatorPressureDrag(double _dt) : dt(_dt)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 1, 6);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }

  ~OperatorPressureDrag() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
  {
    const Real prefactor = -.5 / (info.h_gridpoint);
    pressureDrag[0] = 0;
    pressureDrag[1] = 0;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      pressureDrag[0] += prefactor * (lab(ix+1,iy  ).p - lab(ix-1,iy  ).p);
      pressureDrag[1] += prefactor * (lab(ix  ,iy+1).p - lab(ix  ,iy-1).p);
    }
  }

  inline Real getDrag(int component)
  {
    return pressureDrag[component];
  }
};

template <typename Lab>
class CoordinatorPressure : public GenericCoordinator
{
 protected:
  const double minRho = sim.minRho();

  //#ifndef _MIXED_
  //#ifndef _BOX_
  //#ifndef _OPENBOX_
  PoissonSolverScalarFFTW<FluidGrid> pressureSolver;
  //#else
  //  PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
  //#endif // _OPENBOX_
  //#else
  //  PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
  //#endif // _BOX_
  //#else
  //  PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
  //#endif // _MIXED_

  inline void updatePressure()
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const BlockInfo& info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
        {
          b(ix,iy).pOld = b(ix,iy).p;
          b(ix,iy).p    = b(ix,iy).tmp;
        }
    }
  }

  template <typename Operator>
  void computeSplit(const double dt)
  {
    #pragma omp parallel
    {
      Operator kernel(dt, minRho);
      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(dynamic)
      for(size_t i=0; i<vInfo.size(); i++) {
        const BlockInfo& info = vInfo[i];
        mylab.load(info, 0);
        kernel(mylab, info, *(FluidBlock*)info.ptrBlock);
      }
    }
  }

  inline void drag()
  {
    /*
    const int N = vInfo.size();
    Real tmpDragX = 0, tmpDragY = 0;

    #pragma omp parallel
    {
      OperatorPressureDrag kernel(0);

      Lab mylab;
      mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(static) reduction(+:tmpDragX,tmpDragY)
      for (int i=0; i<N; i++) {
        BlockInfo info = vInfo[i];
        mylab.load(info, 0);
        kernel(mylab, info, *(FluidBlock*)info.ptrBlock);
        tmpDragX += kernel.getDrag(0);
        tmpDragY += kernel.getDrag(1);
      }
    }
    shape->dragP[0] = tmpDragX;
    shape->dragP[1] = tmpDragY;
    */
  }

 public:
  CoordinatorPressure(SimulationData& s) :
  GenericCoordinator(s), pressureSolver(*(s.grid)) { }

  void operator()(const double dt)
  {
    // need an interface that is the same for all solvers - this way the defines can be removed more cleanly

    check("pressure - start");

    computeSplit<OperatorDivergenceSplit>(dt);
    pressureSolver.solve();
    computeSplit<OperatorGradPSplit>(dt);
    updatePressure();

    //drag();

    check("pressure - end");
  }

  string getName()
  {
    return "Pressure";
  }
};
#endif

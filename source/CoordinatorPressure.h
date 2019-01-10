//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "GenericCoordinator.h"

#ifdef CUDAFFT
#include "PoissonSolverScalarCUDA.h"
#define PoissonSolverDCT PoissonSolverPeriodic
#else
#include "PoissonSolverScalarFFTW_freespace.h"
#include "PoissonSolverScalarFFTW_periodic.h"
#include "PoissonSolverScalarFFTW_DCT.h"
#endif

class OperatorDivergenceSplit : public GenericLabOperator {
 private:
  const double dt;
  const Real rho0;
  const PoissonSolverBase * const solver;
  static inline Real mean(const Real a, const Real b) {return .5 * (a + b);}
 public:
  OperatorDivergenceSplit(double _dt, SimulationData& s,
    const PoissonSolverBase * const ps) :
    dt(_dt), rho0(s.minRho()), solver(ps) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 7, 0,1,2,3,4,6,7);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorDivergenceSplit() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    // here i multiply both rhs terms by dt and obtain p*dt from poisson solver
    const Real invH = std::sqrt(dt)/info.h_gridpoint;  // should be 1/h
    const Real factor = rho0 * 0.5 / info.h_gridpoint; // should be [...]/dt
    const size_t offset = solver->_offset(info);
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement& phi  = lab(ix  ,iy  );
      const FluidElement& phiN = lab(ix  ,iy+1);
      const FluidElement& phiS = lab(ix  ,iy-1);
      const FluidElement& phiE = lab(ix+1,iy  );
      const FluidElement& phiW = lab(ix-1,iy  );
      const Real pN = 2*phiN.p - phiN.pOld;
      const Real pS = 2*phiS.p - phiS.pOld;
      const Real pW = 2*phiW.p - phiW.pOld;
      const Real pE = 2*phiE.p - phiE.pOld;
      const Real p  = 2*phi.p  - phi.pOld;
      const Real rN = invH*(1-rho0*mean(phiN.invRho,phi.invRho));
      const Real rS = invH*(1-rho0*mean(phiS.invRho,phi.invRho));
      const Real rE = invH*(1-rho0*mean(phiE.invRho,phi.invRho));
      const Real rW = invH*(1-rho0*mean(phiW.invRho,phi.invRho));
      const Real dE = invH*(pE-p), dW = invH*(p-pW);
      const Real dN = invH*(pN-p), dS = invH*(p-pS);
      const Real divVel =           (phiE.u   -phiW.u    + phiN.v   -phiS.v);
      const Real divDef = phi.tmp * (phiE.tmpU-phiW.tmpU + phiN.tmpV-phiS.tmpV);
      const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
      solver->_cub2fftw(offset, iy, ix, factor*(divVel-divDef) + hatPfac);
    }
  }
};

class OperatorGradPSplit : public GenericLabOperator {
 private:
  const double rho0, dt;
 public:
  OperatorGradPSplit(double _dt, SimulationData& s,
    const PoissonSolverBase * const ps) : rho0(s.minRho()), dt(_dt) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 3, 5,6,7);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorGradPSplit() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real prefactor = -.5 * dt / info.h_gridpoint, invRho0 = 1 / rho0;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement& phi  = lab(ix  ,iy  );
      const FluidElement& phiN = lab(ix  ,iy+1);
      const FluidElement& phiS = lab(ix  ,iy-1);
      const FluidElement& phiE = lab(ix+1,iy  );
      const FluidElement& phiW = lab(ix-1,iy  );
      const Real pN = 2*phiN.p - phiN.pOld, pS = 2*phiS.p - phiS.pOld;
      const Real pW = 2*phiW.p - phiW.pOld, pE = 2*phiE.p - phiE.pOld;
      // tmp contains the pressure correction after the Poisson solver
      o(ix,iy).u += prefactor * (phiE.tmp - phiW.tmp) * invRho0;
      o(ix,iy).v += prefactor * (phiN.tmp - phiS.tmp) * invRho0;
      o(ix,iy).u += prefactor * (pE - pW) * (phi.invRho - invRho0);
      o(ix,iy).v += prefactor * (pN - pS) * (phi.invRho - invRho0);
    }
  }
};

class OperatorDivergence : public GenericLabOperator
{
 private:
  const double dt;
  const PoissonSolverBase * const solver;
  const Real extent[2];
 public:
  OperatorDivergence(double _dt, SimulationData& s,
    const PoissonSolverBase*const ps): dt(_dt), solver(ps),
    extent {s.bpdx/ (Real) std::max(s.bpdx, s.bpdy),
            s.bpdy/ (Real) std::max(s.bpdx, s.bpdy) }
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 4, 0,1,2,3);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorDivergence() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    static constexpr int Z = 8, buffer = 8;
    const Real h = info.h_gridpoint, iWidth = 1/(buffer*h), factor = .5*h/dt;

    const auto _is_touching = [&] (const BlockInfo& i) {
      Real min_pos[2], max_pos[2];
      i.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
      i.pos(min_pos, 0, 0);
      const bool touchN = (Z+buffer)*h >= extent[1] - max_pos[1];
      const bool touchE = (Z+buffer)*h >= extent[0] - max_pos[0];
      const bool touchS = (Z+buffer)*h >= min_pos[1];
      const bool touchW = (Z+buffer)*h >= min_pos[0];
      return touchN || touchE || touchS || touchW;
    };

    const size_t offset = solver->_offset(info);
    if( not _is_touching(info) )
    {
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        const FluidElement& phi  = lab(ix  ,iy  );
        const FluidElement& phiN = lab(ix  ,iy+1);
        const FluidElement& phiS = lab(ix  ,iy-1);
        const FluidElement& phiE = lab(ix+1,iy  );
        const FluidElement& phiW = lab(ix-1,iy  );
        const Real divVel =         (phiE.u   -phiW.u    + phiN.v   -phiS.v);
        const Real divDef = phi.tmp*(phiE.tmpU-phiW.tmpU + phiN.tmpV-phiS.tmpV);
        solver->_cub2fftw(offset, iy, ix, factor*(divVel-divDef));
      }
    }
    else
    {
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        Real p[2]; info.pos(p, ix, iy);
        const FluidElement& phi  = lab(ix  ,iy  );
        const FluidElement& phiN = lab(ix  ,iy+1);
        const FluidElement& phiS = lab(ix  ,iy-1);
        const FluidElement& phiE = lab(ix+1,iy  );
        const FluidElement& phiW = lab(ix-1,iy  );
        const Real divVel =         (phiE.u   -phiW.u    + phiN.v   -phiS.v);
        const Real divDef = phi.tmp*(phiE.tmpU-phiW.tmpU + phiN.tmpV-phiS.tmpV);

        const Real arg1= std::max((Real)0, (Z+buffer)*h -(extent[0]-p[0]) );
        const Real arg2= std::max((Real)0, (Z+buffer)*h -(extent[1]-p[1]) );
        const Real arg3= std::max((Real)0, (Z+buffer)*h -p[0] );
        const Real arg4= std::max((Real)0, (Z+buffer)*h -p[1] );
        const Real dist= std::min(std::max({arg1, arg2, arg3, arg4}), buffer*h);
        //b(ix, iy).s = std::max(1-factor, 1-factor*std::pow(dist*iWidth, 2));
        const Real fade= 1 - std::pow(dist*iWidth, 2);
        solver->_cub2fftw(offset, iy, ix, fade*factor*(divVel-divDef));
      }
    }
  }
};

class OperatorGradP : public GenericLabOperator
{
 private:
  const double dt;
 public:
  OperatorGradP(double _dt, SimulationData& s,
    const PoissonSolverBase*const p): dt(_dt)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 1, 5);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorGradP() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    const Real prefactor = -.5 / info.h_gridpoint * dt;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      // tmp contains the pressure correction after the Poisson solver
      o(ix,iy).u += prefactor * (lab(ix+1,iy).tmp - lab(ix-1,iy).tmp);
      o(ix,iy).v += prefactor * (lab(ix,iy+1).tmp - lab(ix,iy-1).tmp);
      o(ix,iy).p = o(ix,iy).tmp; // copy pressure onto field p
    }
  }
};

template <typename Lab>
class CoordinatorPressure : public GenericCoordinator
{
 protected:
  const double minRho = sim.minRho();
  const int bFS = sim.poissonType, bVariableDensity = sim.bVariableDensity;
  const PoissonSolverBase * const pressureSolver =
   bFS==1? static_cast<PoissonSolverBase*>(new PoissonSolverFreespace(sim.grid))
    : (
   bFS==0? static_cast<PoissonSolverBase*>(new PoissonSolverPeriodic( sim.grid))
         : static_cast<PoissonSolverBase*>(new PoissonSolverDCT( sim.grid ) ) );

  inline void updatePressure() {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      const BlockInfo& info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          b(ix,iy).pOld = b(ix,iy).p;
          b(ix,iy).p = b(ix,iy).tmp;
        }
    }
  }

  template <typename Operator>
  void computeSplit(const double dt) {
    #pragma omp parallel
    {
      Operator K(dt, sim, pressureSolver);
      Lab mylab;
      mylab.prepare(*(sim.grid), K.stencil_start, K.stencil_end, false);
      #pragma omp for schedule(static)
      for(size_t i=0; i<vInfo.size(); i++) {
        const BlockInfo& info = vInfo[i];
        mylab.load(info, 0);
        K(mylab, info, *(FluidBlock*)info.ptrBlock);
      }
    }
  }

 public:
  void operator()(const double dt)
  {
    check("pressure - start");
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy).tmpU = 0; b(ix,iy).tmpV = 0;
      }
    }
    for( const auto& shape : sim.shapes ) shape->deformation_velocities();

    if(bVariableDensity)
    {
      computeSplit<OperatorDivergenceSplit>(dt);
      pressureSolver->solve();
      computeSplit<OperatorGradPSplit>(dt);
      updatePressure();
    }
    else
    {
      computeSplit<OperatorDivergence>(dt);
      pressureSolver->solve();
      computeSplit<OperatorGradP>(dt);
    }
    check("pressure - end");
  }

  CoordinatorPressure(SimulationData& s) : GenericCoordinator(s) {}
  string getName() {
    return "Pressure";
  }
  virtual ~CoordinatorPressure() {
    delete pressureSolver;
  }
};

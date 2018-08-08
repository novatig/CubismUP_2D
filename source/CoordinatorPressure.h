//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "GenericCoordinator.h"
#include "PoissonSolverScalarFFTW_freespace.h"
#include "PoissonSolverScalarFFTW_periodic.h"

class OperatorDivergenceSplit : public GenericLabOperator {
 private:
  const double dt;
  const Real rho0;
  const PoissonSolverBase * const solver;
  static inline Real mean(const Real a, const Real b) {return .5 * (a + b);}
 public:
  OperatorDivergenceSplit(double _dt, double _rho0,
    const PoissonSolverBase * const ps) :
    dt(_dt), rho0(_rho0), solver(ps) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 7, 0,1,2,3,4,6,7);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorDivergenceSplit() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    const Real invH = 1.0/info.h_gridpoint;
    const Real factor = rho0 * 0.5/(info.h_gridpoint * dt);
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
  OperatorGradPSplit(double _dt, double _rho0,
    const PoissonSolverBase * const ps) : rho0(_rho0), dt(_dt) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 3, 5,6,7);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorGradPSplit() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    const double dh = info.h_gridpoint;
    const Real prefactor = -.5 * dt / dh;
    const Real invRho0 = 1/rho0;
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
      o(ix,iy).u += prefactor*invRho0 * (phiE.tmp - phiW.tmp);
      o(ix,iy).v += prefactor*invRho0 * (phiN.tmp - phiS.tmp);
      o(ix,iy).u += prefactor * (pE - pW) * (phi.invRho - invRho0);
      o(ix,iy).v += prefactor * (pN - pS) * (phi.invRho - invRho0);
    }
  }
};

class OperatorDivergence : public GenericLabOperator {
 private:
  const double dt;
  const PoissonSolverBase * const solver;
 public:
  OperatorDivergence(double _dt,double _rho0,const PoissonSolverBase*const ps):
    dt(_dt), solver(ps) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 4, 0,1,2,3);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorDivergence() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    const Real factor = 0.5/(info.h_gridpoint * dt);
    const size_t offset = solver->_offset(info);
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement& phi  = lab(ix  ,iy  );
      const FluidElement& phiN = lab(ix  ,iy+1);
      const FluidElement& phiS = lab(ix  ,iy-1);
      const FluidElement& phiE = lab(ix+1,iy  );
      const FluidElement& phiW = lab(ix-1,iy  );
      const Real divVel =           (phiE.u   -phiW.u    + phiN.v   -phiS.v);
      const Real divDef = phi.tmp * (phiE.tmpU-phiW.tmpU + phiN.tmpV-phiS.tmpV);
      solver->_cub2fftw(offset, iy, ix, factor*(divVel-divDef));
    }
  }
};

class OperatorGradP : public GenericLabOperator {
 private:
  const double dt;
 public:
  OperatorGradP(double _dt,double _r,const PoissonSolverBase*const p): dt(_dt) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 1, 5);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorGradP() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    const Real prefactor = -.5 * dt / info.h_gridpoint;
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
  const bool bFS = sim.bFreeSpace, bVariableDensity = sim.bVariableDensity;
  const PoissonSolverBase * const pressureSolver =
   bFS? static_cast<PoissonSolverBase*>(new PoissonSolverFreespace(sim.grid))
      : static_cast<PoissonSolverBase*>(new PoissonSolverPeriodic( sim.grid));

  inline void updatePressure() {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      const BlockInfo& info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          b(ix,iy).pOld = b(ix,iy).p;
          b(ix,iy).p    = b(ix,iy).tmp;
        }
    }
  }

  template <typename Operator>
  void computeSplit(const double dt) {
    #pragma omp parallel
    {
      Operator K(dt, minRho, pressureSolver);
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

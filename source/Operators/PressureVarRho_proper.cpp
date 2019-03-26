//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureVarRho_proper.h"
#include "../Poisson/HYPREdirichletVarRho.h"

using namespace cubism;

void PressureVarRho_proper::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0);
      const ScalarLab  &__restrict__   P = presLab;
            VectorBlock&__restrict__   V = *(VectorBlock*) velInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*)iRhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] += pFac*IRHO(ix,iy).s*(P(ix+1,iy).s - P(ix-1,iy).s);
        V(ix,iy).u[1] += pFac*IRHO(ix,iy).s*(P(ix,iy+1).s - P(ix,iy-1).s);
      }
    }
  }
}

void PressureVarRho_proper::operator()(const double dt)
{
  #ifdef HYPREFFT
    const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
    pressureSolver->solve(presInfo, presInfo);
  #else
    printf("Class PressureVarRho_proper REQUIRES HYPRE\n");
    fflush(0); abort();
  #endif

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureVarRho_proper::PressureVarRho_proper(SimulationData& s) : Operator(s)
#ifdef HYPREFFT
  , pressureSolver( new HYPREdirichletVarRho(s) )
#endif
  { }

PressureVarRho_proper::~PressureVarRho_proper()
{
  #ifdef HYPREFFT
    delete pressureSolver;
  #endif
}

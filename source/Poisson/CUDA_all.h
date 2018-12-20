//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "PoissonSolver.h"

#include <cufft.h>
#ifndef _FLOAT_PRECISION_
  #define cufftCmpT cufftDoubleComplex
  #define cufftPlanFWD CUFFT_D2Z
  #define cufftPlanBWD CUFFT_Z2D
#else //_FLOAT_PRECISION_
  #define cufftCmpT cufftComplex
  #define cufftPlanFWD CUFFT_R2C
  #define cufftPlanBWD CUFFT_C2R
#endif//_FLOAT_PRECISION_

void freeCuMem(Real * buf);
void clearCuMem(Real * buf, const size_t size);
void allocCuMem(Real* & ptr, const size_t size);
void freePlan(cufftHandle& plan);
void makePlan(cufftHandle& handle, const int mx, const int my, cufftType plan);
void dPeriodic(const cufftHandle&fwd, const cufftHandle&bwd, const int mx,
  const int my, const Real h, Real*const rhs, Real*const rhs_gpu);
void dFreespace(const cufftHandle&fwd, const cufftHandle&bwd, const int nx,
  const int ny, Real*const rhs, const Real*const G_hat, Real*const rhs_gpu);
void initGreen(const int nx, const int ny, const Real h, Real*const m_kernel);

class CUDA_periodic : public PoissonSolver
{
 protected:
  const size_t MX = totNx, MY = totNy, MX_hat = MX/2 +1;
  const Real facX = 2.0*M_PI/MX, facY = 2.0*M_PI/MY, norm = 1.0/(MY*MX);

  cufftHandle fwd, bwd;
  Real * rhs_gpu = nullptr;

 public:

  #define TOT_DOF_X s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX
  #define STRIDE 2 * ( (TOT_DOF_X)/2 +1 )

  CUDA_periodic(SimulationData& s) : PoissonSolver(s, STRIDE)
  {
    makePlan(fwd, MY, MX, cufftPlanFWD);
    makePlan(bwd, MY, MX, cufftPlanBWD);
    assert(2*sizeof(Real) == sizeof(cufftCmpT));
    buffer = (Real*) malloc( MY * MX_hat * 2 * sizeof(Real) );
    allocCuMem(rhs_gpu, MY * MX_hat * sizeof(cufftCmpT) );
  }
  #undef TOT_DOF_X
  #undef STRIDE

  void solve() override {
    cub2rhs();
    dPeriodic(fwd, bwd, MY, MX, sim.getH(), buffer, rhs_gpu);
    sol2cub();
  }

  ~CUDA_periodic() {
    freePlan(fwd);
    freePlan(bwd);
    freeCuMem(rhs_gpu);
    free(buffer);
  }
};

class CUDA_freespace : public PoissonSolver
{
  const size_t MX = 2*totNx - 1;
  const size_t MY = 2*totNy - 1;
  const size_t MX_hat = MX/2 +1;
  const Real norm_factor = 1.0/(MY*MX);

  cufftHandle fwd, bwd;
  Real * rhs_gpu = nullptr;
  Real * m_kernel = nullptr;

 public:
  #define TOT_DOF_X s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX
  CUDA_freespace(SimulationData& s) : PoissonSolver(s, TOT_DOF_X)
  {
    makePlan(fwd, MY, MX, cufftPlanFWD);
    makePlan(bwd, MY, MX, cufftPlanBWD);
    assert(2*sizeof(Real) == sizeof(cufftCmpT));
    buffer = (Real*) malloc(MY * MX_hat * 2 * sizeof(Real) );
    allocCuMem(rhs_gpu,  MY * MX_hat * sizeof(cufftCmpT) );
    clearCuMem(rhs_gpu, MY * MX_hat * sizeof(cufftCmpT) );
    allocCuMem(m_kernel, MY * MX_hat * sizeof(Real) );
    initGreen(totNy, totNx, sim.getH(), m_kernel);
  }
  #undef TOT_DOF_X

  void solve()  override {
    cub2rhs();
    dFreespace(fwd, bwd, totNy, totNx, buffer, m_kernel, rhs_gpu);
    sol2cub();
  }

  ~CUDA_freespace() {
    freePlan(fwd);
    freePlan(bwd);
    freeCuMem(rhs_gpu);
    freeCuMem(m_kernel);
    free(buffer);
  }
};

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include <PoissonSolverScalar.h>

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

class PoissonSolverPeriodic : public PoissonSolverBase
{
 protected:
  const int mx = nx;
  const int my = ny;
  const int my_hat = my/2 +1;
  const Real facX = 2.0*M_PI/(mx*h), facY = 2.0*M_PI/(my*h), norm = 1./(mx*my);

  cufftHandle fwd, bwd;
  Real * rhs_gpu = nullptr;

 public:

  PoissonSolverPeriodic(FluidGrid*const _grid) : PoissonSolverBase(*_grid, 0)
  {
    makePlan(fwd, mx, my, cufftPlanFWD);
    makePlan(bwd, mx, my, cufftPlanBWD);
    assert(2*sizeof(Real) == sizeof(cufftCmpT));
    rhs = (Real*) malloc(mx * my_hat * 2 * sizeof(Real) );
    allocCuMem(rhs_gpu, mx * my_hat * sizeof(cufftCmpT) );
  }

  void solve() const override {
    dPeriodic(fwd, bwd, mx, my, h, rhs, rhs_gpu);
    _fftw2cub();
  }

  ~PoissonSolverPeriodic() {
    freePlan(fwd);
    freePlan(bwd);
    freeCuMem(rhs_gpu);
    free(rhs);
  }
};

class PoissonSolverFreespace : public PoissonSolverBase
{
 protected:
  cufftHandle fwd, bwd;
  Real * rhs_gpu = nullptr;
  Real * m_kernel = nullptr;

 public:

  PoissonSolverFreespace(FluidGrid*const _grid) : PoissonSolverBase(*_grid, 0)
  {
    const int ny_hat = ny/2 +1;
    assert((size_t)ny_hat == my_hat);
    const int Mx = 2 * nx - 1;
    const int My = 2 * ny - 1;
    const int My_hat = My/2 +1;
    makePlan(fwd, Mx, My, cufftPlanFWD);
    makePlan(bwd, Mx, My, cufftPlanBWD);
    assert(2*sizeof(Real) == sizeof(cufftCmpT));
    rhs = (Real*) malloc(nx * ny_hat * 2 * sizeof(Real) );
    allocCuMem(rhs_gpu,  Mx * My_hat * sizeof(cufftCmpT) );
    clearCuMem(rhs_gpu, Mx * My_hat * sizeof(cufftCmpT) );
    allocCuMem(m_kernel, Mx * My_hat * sizeof(Real) );
    initGreen(nx, ny, h, m_kernel);
  }

  void solve() const override {
    dFreespace(fwd, bwd, nx, ny, rhs, m_kernel, rhs_gpu);
    _fftw2cub();
  }

  ~PoissonSolverFreespace() {
    freePlan(fwd);
    freePlan(bwd);
    freeCuMem(rhs_gpu);
    freeCuMem(m_kernel);
    free(rhs);
  }
};

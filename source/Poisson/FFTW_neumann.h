//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "PoissonSolver.h"

#include <fftw3.h>
#ifndef _FLOAT_PRECISION_
typedef fftw_complex mycomplex;
typedef fftw_plan myplan;
#else // _FLOAT_PRECISION_
typedef fftwf_complex mycomplex;
typedef fftwf_plan myplan;
#endif // _FLOAT_PRECISION_

class FFTW_neumann : public PoissonSolver
{
  const size_t MX = totNx, MY = totNy;
  float * const COScoefX = new float[MX];
  float * const COScoefY = new float[MY];
  const Real norm_factor = 0.25/(MX*MY);
  myplan fwd, bwd;

  inline void _solve() const
  {
    const Real waveFactX = M_PI/MX, waveFactY = M_PI/MY;
    Real * __restrict__ const in_out = buffer;
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j)
    for(size_t i=0; i<MX; ++i) {
      const Real rkx = (i+(Real).5)*waveFactX, rky = (j+(Real).5)*waveFactY;
      in_out[j * MX + i] *= - norm_factor / (rkx*rkx+rky*rky);
    }
    in_out[0] = 0; //this is sparta! (part 2)
  }

  inline void _solveSpectral() const
  {
    const Real waveFactX = M_PI/MX, waveFactY = M_PI/MY;
    Real * __restrict__ const in_out = buffer;
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j)
    for(size_t i=0; i<MX; ++i) {
      const Real rkx = (i + (Real).5)*waveFactX, rky = (j + (Real).5)*waveFactY;
      const Real denomFD = 1 - COScoefX[i]/2 - COScoefY[j]/2;
      const Real denomSP = rkx*rkx + rky*rky;
      in_out[j * MX + i] *=  - norm_factor / ( 0.9*denomFD + 0.1*denomSP );
    }
    in_out[0] = 0; //this is sparta! (part 2)
    ///*
    //in_out[    MX-1 ] = 0; // j=0, i=end
    //in_out[MX*(MY-1)] = 0; // j=end, i=0
    //in_out[MX*MY -1 ] = 0; // j=end, i=end
    //*/
    /*
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j) in_out[   j   * MX + (MX-1)] = 0;
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j) in_out[   j   * MX + (MX-2)] = 0;
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<MX; ++i) in_out[(MY-1) * MX +    i  ] = 0;
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<MX; ++i) in_out[(MY-2) * MX +    i  ] = 0;
    */
  }

 public:

  #define TOT_DOF_X s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX

  FFTW_neumann(SimulationData& s) : PoissonSolver(s, TOT_DOF_X)
  {
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j) COScoefY[j] = std::cos(M_PI/MY*2.0*j);
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<MX; ++i) COScoefX[i] = std::cos(M_PI/MX*2.0*i);

    printf("Employing FFTW-based Poisson solver by cosine transform.\n");
    const int desired_threads = omp_get_max_threads();
    #ifndef _FLOAT_PRECISION_
      const int retval = fftw_init_threads();
      fftw_plan_with_nthreads(desired_threads);
      buffer = fftw_alloc_real(MY * MX);
      fwd = fftw_plan_r2r_2d(MY, MX, buffer, buffer, FFTW_RODFT10, FFTW_RODFT10, FFTW_MEASURE);
      bwd = fftw_plan_r2r_2d(MY, MX, buffer, buffer, FFTW_RODFT01, FFTW_RODFT01, FFTW_MEASURE);
    #else // _FLOAT_PRECISION_
      const int retval = fftwf_init_threads();
      fftwf_plan_with_nthreads(desired_threads);
      buffer = fftwf_alloc_real(MY * MX);
      fwd = fftwf_plan_r2r_2d(MY, MX, buffer, buffer, FFTW_RODFT10, FFTW_RODFT10, FFTW_MEASURE);
      bwd = fftwf_plan_r2r_2d(MY, MX, buffer, buffer, FFTW_RODFT01, FFTW_RODFT01, FFTW_MEASURE);
    #endif // _FLOAT_PRECISION_
    if(retval==0) {
      std::cout<<"Call to fftw_init_threads() returned zero. Aborting\n";
      abort();
    }
  }

  #undef TOT_DOF_X

  void solve(const std::vector<cubism::BlockInfo>& BSRC,
             const std::vector<cubism::BlockInfo>& BDST) override
  {
    sim.startProfiler("FFTW_cub2fft");
    cub2rhs(BSRC);
    sim.stopProfiler();

    sim.startProfiler("FFTW_bwd");
    #ifndef _FLOAT_PRECISION_
      fftw_execute(fwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(fwd);
    #endif // _FLOAT_PRECISION_
    sim.stopProfiler();

    sim.startProfiler("FFTW_solve");
      _solve();
      //_solveSpectral();
    sim.stopProfiler();

    sim.startProfiler("FFTW_fwd");
    #ifndef _FLOAT_PRECISION_
      fftw_execute(bwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(bwd);
    #endif // _FLOAT_PRECISION_
    sim.stopProfiler();

    sim.startProfiler("FFTW_fft2cub");
    sol2cub(BDST);
    sim.stopProfiler();
  }

  ~FFTW_neumann()
  {
    delete [] COScoefX;
    delete [] COScoefY;
    #ifndef _FLOAT_PRECISION_
      fftw_cleanup_threads();
      fftw_destroy_plan(fwd);
      fftw_destroy_plan(bwd);
      fftw_free(buffer);
    #else // _FLOAT_PRECISION_
      fftwf_cleanup_threads();
      fftwf_destroy_plan(fwd);
      fftwf_destroy_plan(bwd);
      fftwf_free(buffer);
    #endif // _FLOAT_PRECISION_
  }
};

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

class FFTW_periodic : public PoissonSolver
{
  const size_t MX = totNx, MY = totNy, MX_hat = MX/2 +1;
  const Real norm_factor = 1.0/(MX*MY);
  myplan fwd, bwd;

 protected:

  void _solve_finiteDiff()
  {
    mycomplex * __restrict__ const in_out = (mycomplex *) buffer;
    const Real waveFactX = 2*M_PI/MX, waveFactY = 2*M_PI/MY;
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j)
    for(size_t i=0; i<MX_hat; ++i)
    {
      const size_t kx = i<=MX/2 ? i : MX-i, ky = j<=MY/2 ? j : MY-j;
      #if 0 // based on the 5 point stencil in 1D (h^4 error)
        const Real cosx = std::cos(waveFactX*kx), cosy = std::cos(waveFactY*ky);
        const Real denom = 32*(cosx + cosy) - 4*(cosx*cosx + cosy*cosy) - 56;
        //const Real X = waveFactX*i, Y = waveFactY * j;
        //const Real denom = 32*(std::cos(X) + std::cos(Y))
        //                  - 2*(std::cos(2*X) + std::cos(2*Y)) - 60;
        const Real inv_denom = denom == 0 ? 0 : 1 / denom;
        const Real fatfactor = 12 * norm_factor * inv_denom;
      #elif 1 // based on the 3 point stencil in 1D (2h^2 error)
        const Real cosx = std::cos(2*waveFactX*kx);
        const Real cosy = std::cos(2*waveFactY*ky);
        const Real fatfactor = norm_factor / ( cosx/2 + cosy/2 - 1 );
      #elif 1 // based on the 3 point stencil in 1D (h^2 error)
        const Real cosx = std::cos(waveFactX*kx), cosy = std::cos(waveFactY*ky);
        const Real denom =  2*cosx + 2*cosy - 4;
        const Real fatfactor = norm_factor / denom;
      #else // this is to check the transform only
        const Real fatfactor = norm_factor;
      #endif
      in_out[j*MX_hat + i][0] *= fatfactor;
      in_out[j*MX_hat + i][1] *= fatfactor;
    }
    in_out[0][0] = 0; in_out[0][1] = 0; //this is sparta!
    #if 1
    in_out[MX/2][0] = 0;              in_out[MX/2][1] = 0;
    in_out[MY/2*MX_hat][0] = 0;       in_out[MY/2*MX_hat][1] = 0;
    in_out[MY/2*MX_hat+ MX/2][0] = 0; in_out[MY/2*MX_hat + MX/2][1] = 0;
    #endif
  }

  inline void _solve_spectral() const
  {
    mycomplex * __restrict__ const in_out = (mycomplex *) buffer;
    const Real waveFactX = 2*M_PI/MX, waveFactY = 2*M_PI/MY;
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j)
    for(size_t i=0; i<MX_hat; ++i)
    {
      const size_t kx = i <= MX/2 ? i : MX-i, ky = j <= MY/2 ? j : MY-j;
      const Real rkx = kx*waveFactX, rky = ky*waveFactY;
      const Real kinv = -1/(rkx*rkx+rky*rky); //this is sparta! (part 1)
      in_out[j*MX_hat + i][0] *= kinv*norm_factor;
      in_out[j*MX_hat + i][1] *= kinv*norm_factor;
    }
    in_out[0][0] = 0; in_out[0][1] = 0; //this is sparta! (part 2)
  }

public:

  #define TOT_DOF_X s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX
  #define STRIDE 2 * ( (TOT_DOF_X)/2 +1 )

  FFTW_periodic(SimulationData& s) : PoissonSolver(s, STRIDE)
  {
    printf("Employing FFTW-based Poisson solver by Fourier transform.\n");
    const int desired_threads = omp_get_max_threads();
    #ifndef _FLOAT_PRECISION_
      const int retval = fftw_init_threads();
      fftw_plan_with_nthreads(desired_threads);
      buffer = fftw_alloc_real(2 * MY * MX_hat);
      fwd = fftw_plan_dft_r2c_2d(MY,MX,buffer,(mycomplex *)buffer,FFTW_MEASURE);
      bwd = fftw_plan_dft_c2r_2d(MY,MX,(mycomplex *)buffer,buffer,FFTW_MEASURE);
    #else // _FLOAT_PRECISION_
      const int retval = fftwf_init_threads();
      fftwf_plan_with_nthreads(desired_threads);
      buffer = fftwf_alloc_real(2 * MY * MX_hat);
      fwd =fftwf_plan_dft_r2c_2d(MY,MX,buffer,(mycomplex *)buffer,FFTW_MEASURE);
      bwd =fftwf_plan_dft_c2r_2d(MY,MX,(mycomplex *)buffer,buffer,FFTW_MEASURE);
    #endif // _FLOAT_PRECISION_
    if(retval==0) {
      std::cout<<"Call fftw_init_threads() returned zero.\n"; fflush(0);abort();
    }
  }

  #undef TOT_DOF_X
  #undef STRIDE

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
      //_solve_spectral();
      _solve_finiteDiff();
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

  ~FFTW_periodic()
  {
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

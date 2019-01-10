#pragma once

#include <PoissonSolverScalar.h>

#include <fftw3.h>
#ifndef _FLOAT_PRECISION_
typedef fftw_complex mycomplex;
typedef fftw_plan myplan;
#else // _FLOAT_PRECISION_
typedef fftwf_complex mycomplex;
typedef fftwf_plan myplan;
#endif // _FLOAT_PRECISION_

class PoissonSolverDCT : public PoissonSolverBase
{
  const size_t mx = nx;
  const size_t my = ny;
  const Real norm_factor = 0.25/(nx*ny);

  myplan fwd, bwd;

 protected:


  inline void _solve() const
  {
    const Real waveFactX = 1.0*M_PI/nx;
    const Real waveFactY = 1.0*M_PI/ny;
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<my; ++j) {
        const int linidx = i * my + j;
        const Real rkx = (i+0.5)*waveFactX;
        const Real rky = (j+0.5)*waveFactY;
        //const Real kinv = (kx==0 && ky==0) ? 0 : -1/(rkx*rkx+rky*rky);
        const Real kinv = -1/(rkx*rkx+rky*rky); //this is sparta! (part 1)
        rhs[linidx] *= kinv*norm_factor;
      }
    rhs[0] = 0; //this is sparta! (part 2)
  }

public:

  PoissonSolverDCT(FluidGrid*const _grid) :
  PoissonSolverBase(*_grid,0,_grid->getBlocksPerDimension(0)*BlockType::sizeX/2)
  {
    const int desired_threads = omp_get_max_threads();
    #ifndef _FLOAT_PRECISION_
      const int retval = fftw_init_threads();
      fftw_plan_with_nthreads(desired_threads);
      rhs = fftw_alloc_real(mx * my);
      fwd = fftw_plan_r2r_2d(mx, my, rhs, rhs, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
      bwd = fftw_plan_r2r_2d(mx, my, rhs, rhs, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
    #else // _FLOAT_PRECISION_
      const int retval = fftwf_init_threads();
      fftwf_plan_with_nthreads(desired_threads);
      rhs = fftwf_alloc_real(mx * my);
      fwd = fftwf_plan_r2r_2d(mx, my, rhs, rhs, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
      bwd = fftwf_plan_r2r_2d(mx, my, rhs, rhs, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
    #endif // _FLOAT_PRECISION_
    if(retval==0) {
      cout << "FFTWBase::setup(): Oops the call to fftw_init_threads() returned zero. Aborting\n";
      abort();
    }
  }

  void solve() const override
  {
    #ifndef _FLOAT_PRECISION_
      fftw_execute(fwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(fwd);
    #endif // _FLOAT_PRECISION_

    _solve();

    #ifndef _FLOAT_PRECISION_
      fftw_execute(bwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(bwd);
    #endif // _FLOAT_PRECISION_
    _fftw2cub();
  }

  ~PoissonSolverDCT() {
    #ifndef _FLOAT_PRECISION_
      fftw_cleanup_threads();
      fftw_destroy_plan(fwd);
      fftw_destroy_plan(bwd);
      fftw_free(rhs);
    #else // _FLOAT_PRECISION_
      fftwf_cleanup_threads();
      fftwf_destroy_plan(fwd);
      fftwf_destroy_plan(bwd);
      fftwf_free(rhs);
    #endif // _FLOAT_PRECISION_
  }
};

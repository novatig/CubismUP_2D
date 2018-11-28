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

class PoissonSolverPeriodic : public PoissonSolverBase
{
  const size_t mx = nx;
  const size_t my = ny;
  const size_t my_hat = my/2 +1;
  const Real norm_factor = 1./(nx*ny);

  myplan fwd, bwd;

 protected:

  void _solve_finiteDiff()
  {
    const Real h2 = h*h;
    const Real factor = h2*norm_factor;
    mycomplex * const in_out = (mycomplex *)lhs;

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<my_hat; ++j)
      {
        const int linidx = i*my_hat+j;

        // based on the 5 point stencil in 1D (h^4 error)
        const Real denom = 32.*(cos(2.*M_PI*i/nx) + cos(2.*M_PI*j/ny)) - 2.*(cos(4.*M_PI*i/nx) + cos(4.*M_PI*j/ny)) - 60.;
        const Real inv_denom = (denom==0)? 0.:1./denom;
        const Real fatfactor = 12. * inv_denom * factor;

        // based on the 3 point stencil in 1D (h^2 error)
        //const Real denom = 2.*(cos(2.*M_PI*i/nx) + cos(2.*M_PI*j/ny)) - 4.;
        //const Real inv_denom = (denom==0)? 0.:1./denom;
        //const Real fatfactor = inv_denom * factor;

        // this is to check the transform only
        //const Real fatfactor = norm_factor;

        in_out[linidx][0] *= fatfactor;
        in_out[linidx][1] *= fatfactor;
      }

    //this is sparta!
    in_out[0][0] = 0; in_out[0][1] = 0;
  }

  inline void _solve() const
  {
    mycomplex * const in_out = (mycomplex *)lhs;

    const Real waveFactX = 2.0*M_PI/(nx*h);
    const Real waveFactY = 2.0*M_PI/(ny*h);
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<my_hat; ++j) {
        const int linidx = i * my_hat + j;
        const int kx = (i <= nx/2) ? i : -(nx-i);
        const int ky = (j <= ny/2) ? j : -(ny-j);
        const Real rkx = kx*waveFactX;
        const Real rky = ky*waveFactY;
        //const Real kinv = (kx==0 && ky==0) ? 0 : -1/(rkx*rkx+rky*rky);
        const Real kinv = -1/(rkx*rkx+rky*rky); //this is sparta! (part 1)
        in_out[linidx][0] *= kinv*norm_factor;
        in_out[linidx][1] *= kinv*norm_factor;
      }
    in_out[0][0] = 0; in_out[0][1] = 0; //this is sparta! (part 2)
  }

public:

  PoissonSolverPeriodic(FluidGrid*const _grid) : PoissonSolverBase(*_grid, 0)
  {
    const int desired_threads = omp_get_max_threads();
    #ifndef _FLOAT_PRECISION_
      const int retval = fftw_init_threads();
      fftw_plan_with_nthreads(desired_threads);
      rhs = fftw_alloc_real(2 * mx * my_hat);
      lhs = fftw_alloc_real(2 * mx * my_hat);
      fwd = fftw_plan_dft_r2c_2d(mx, my, rhs, (mycomplex *)lhs, FFTW_MEASURE);
      bwd = fftw_plan_dft_c2r_2d(mx, my, (mycomplex *)lhs, lhs, FFTW_MEASURE);
    #else // _FLOAT_PRECISION_
      const int retval = fftwf_init_threads();
      fftwf_plan_with_nthreads(desired_threads);
      rhs = fftwf_alloc_real(2 * mx * my_hat);
      lhs = fftwf_alloc_real(2 * mx * my_hat);
      fwd = fftwf_plan_dft_r2c_2d(mx, my, rhs, (mycomplex *)lhs, FFTW_MEASURE);
      bwd = fftwf_plan_dft_c2r_2d(mx, my, (mycomplex *)lhs, lhs, FFTW_MEASURE);
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

  ~PoissonSolverPeriodic() {
    #ifndef _FLOAT_PRECISION_
      fftw_cleanup_threads();
      fftw_destroy_plan(fwd);
      fftw_destroy_plan(bwd);
      fftw_free(rhs);
      fftw_free(lhs);
    #else // _FLOAT_PRECISION_
      fftwf_cleanup_threads();
      fftwf_destroy_plan(fwd);
      fftwf_destroy_plan(bwd);
      fftwf_free(rhs);
      fftwf_free(lhs);
    #endif // _FLOAT_PRECISION_
  }
};

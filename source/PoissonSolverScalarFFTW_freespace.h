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

class PoissonSolverFreespace: public PoissonSolverBase
{
  Real * m_kernel = nullptr;

  myplan fwd, bwd;
 protected:

  void _init_green()
  {
    #ifndef _FLOAT_PRECISION_
      Real* tmp = fftw_alloc_real(2 * mx * my_hat);
      m_kernel = fftw_alloc_real(mx * my_hat);
    #else // _FLOAT_PRECISION_
      Real* tmp = fftwf_alloc_real(2 * mx * my_hat);
      m_kernel = fftwf_alloc_real(mx * my_hat);
    #endif // _FLOAT_PRECISION_
    // This algorithm requires m_size >= 2 and m_N0 % m_size == 0

    // This factor is due to the discretization of the convolution
    // integtal.  It is composed of (h*h) * (-1/[4*pi*h]), where h is the
    // uniform grid spacing.  The first factor is the discrete volume
    // element of the convolution integral; the second factor belongs to
    // Green's function on a uniform mesh.
    // 3D:
    //h * h * h * - 1/(4*pi*h) / ( r>0 ? 1/r : 1 )
    // 2D:
    //h * h     *   1/ (2*pi)  * ( log(h) + log(r)
    //h * h     *   1/ (2*pi)  * ( log(h) +(r>0 ? log(r) : 1)
    //h * h     *   1/ (2*pi)  * ( log(h * max(r, 1) )
    const Real fac = h * h / ( 2*M_PI );
    #pragma omp parallel for
    for (size_t i = 0; i < mx; ++i)
    for (size_t j = 0; j < my; ++j) {
        const Real xi = i>=nx? mx-i : i;
        const Real yi = j>=ny? my-j : j;
        const Real r = std::sqrt(xi*xi + yi*yi);
        const size_t idx = j + 2*my_hat*i;
        if(r > 0) tmp[idx] = fac * std::log(h * r);
        // r_eq = h / sqrt(pi)
        // G = 1/4 * r_eq^2 * (2* ln(r_eq) - 1)
        else      tmp[idx] = fac/2 * (2*std::log(h/std::sqrt(M_PI)) - 1);
    }

    #ifndef _FLOAT_PRECISION_
     myplan tmpp =fftw_plan_dft_r2c_2d(mx,my,tmp,(mycomplex*)tmp,FFTW_MEASURE);
     fftw_execute(tmpp);
     fftw_destroy_plan(tmpp);
    #else // _FLOAT_PRECISION_
     myplan tmpp =fftwf_plan_dft_r2c_2d(mx,my,tmp,(mycomplex*)tmp,FFTW_MEASURE);
     fftwf_execute(tmpp);
     fftwf_destroy_plan(tmpp);
    #endif // _FLOAT_PRECISION_


    const mycomplex *const G_hat = (mycomplex *) tmp;
    const Real norm_factor = 1.0 / (mx * my);
    #pragma omp parallel for
    for (size_t i = 0; i < mx; ++i)
    for (size_t j = 0; j < my_hat; ++j) {
        const size_t linidx = j + my_hat*i;
        m_kernel[linidx] = G_hat[linidx][0] * norm_factor; // real part only
    }

    #ifndef _FLOAT_PRECISION_
      fftw_free(tmp);
    #else // _FLOAT_PRECISION_
      fftwf_free(tmp);
    #endif
  }

 public:
  PoissonSolverFreespace(FluidGrid*const _grid) : PoissonSolverBase(*_grid, 1)
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
      cout << "Call fftw_init_threads() returned zero.\n"; fflush(0); abort();
    }
    _init_green();
    memset(rhs, 0, 2 * mx * my_hat * sizeof(Real));
  }

  void solve() const override
  {
    //const double t0 = omp_get_wtime();
    #ifndef _FLOAT_PRECISION_
      fftw_execute(fwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(fwd);
    #endif // _FLOAT_PRECISION_
    //const double t1 = omp_get_wtime();

    mycomplex * const in_out = (mycomplex *)lhs;
    const Real* const G_hat = m_kernel;
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<mx;     ++i)
    for(size_t j=0; j<my_hat; ++j) {
      const int linidx = i * my_hat + j;
      in_out[linidx][0] *= G_hat[linidx];
      in_out[linidx][1] *= G_hat[linidx];
    }

    //const double t2 = omp_get_wtime();
    #ifndef _FLOAT_PRECISION_
      fftw_execute(bwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(bwd);
    #endif // _FLOAT_PRECISION_
    //const double t3 = omp_get_wtime();

    //printf("UP:%f S:%f DW:%f\n", t1-t0, t2-t1, t3-t2);
    _fftw2cub();
    //memset(rhs, 0, 2 * mx * my_hat * sizeof(Real));
  }

  ~PoissonSolverFreespace() {
    #ifndef _FLOAT_PRECISION_
      fftw_free(m_kernel);
      fftw_cleanup_threads();
      fftw_destroy_plan(fwd);
      fftw_destroy_plan(bwd);
      fftw_free(rhs);
      fftw_free(lhs);
    #else // _FLOAT_PRECISION_
      fftwf_free(m_kernel);
      fftwf_cleanup_threads();
      fftwf_destroy_plan(fwd);
      fftwf_destroy_plan(bwd);
      fftwf_free(rhs);
      fftwf_free(lhs);
    #endif // _FLOAT_PRECISION_
  }
};

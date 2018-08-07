#pragma once

#include <PoissonSolverScalarFFTW.h>

class PoissonSolverFreespace: public PoissonSolverBase
{
  Real * m_kernel = nullptr;

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

  void _solve() const override
  {
    mycomplex * const in_out = (mycomplex *)rhs;
    const Real* const G_hat = m_kernel;

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<mx;     ++i)
    for(size_t j=0; j<my_hat; ++j) {
      const int linidx = i * my_hat + j;
      in_out[linidx][0] *= G_hat[linidx];
      in_out[linidx][1] *= G_hat[linidx];
    }
  }

 public:
  PoissonSolverFreespace(FluidGrid*const _grid) :
    PoissonSolverBase(*_grid, true) {
    _init_green();
  }
  ~PoissonSolverFreespace() {
    #ifndef _FLOAT_PRECISION_
    fftw_free(m_kernel);
    #else // _FLOAT_PRECISION_
    fftwf_free(m_kernel);
    #endif // _FLOAT_PRECISION_
  }
};

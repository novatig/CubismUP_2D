
#include "FFTW_freespace.h"

void FFTW_freespace::solve()
{
  sim.startProfiler("FFTW_cub2fft");
  cub2rhs();
  sim.stopProfiler();

  sim.startProfiler("FFTW_fwd");
  #ifndef _FLOAT_PRECISION_
    fftw_execute(fwd);
  #else // _FLOAT_PRECISION_
    fftwf_execute(fwd);
  #endif // _FLOAT_PRECISION_
  sim.stopProfiler();

  sim.startProfiler("FFTW_solve");
  {
    mycomplex * __restrict__ const in_out = (mycomplex *) buffer;
    const Real* __restrict__ const G_hat = m_kernel;
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<MY; ++j)
    for(size_t i=0; i<MX_hat; ++i) {
      const size_t linidx = j*MX_hat + i;
      in_out[linidx][0] *= G_hat[linidx];
      in_out[linidx][1] *= G_hat[linidx];
    }
  }
  sim.stopProfiler();

  sim.startProfiler("FFTW_bwd");
  #ifndef _FLOAT_PRECISION_
    fftw_execute(bwd);
  #else // _FLOAT_PRECISION_
    fftwf_execute(bwd);
  #endif // _FLOAT_PRECISION_
  sim.stopProfiler();

  sim.startProfiler("FFTW_fft2cub");
  sol2cub();
  memset(buffer, 0, 2 * MY * MX_hat * sizeof(Real));
  sim.stopProfiler();
}

FFTW_freespace::~FFTW_freespace()
{
  #ifndef _FLOAT_PRECISION_
    fftw_free(m_kernel);
    fftw_cleanup_threads();
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(buffer);
  #else // _FLOAT_PRECISION_
    fftwf_free(m_kernel);
    fftwf_cleanup_threads();
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    fftwf_free(buffer);
  #endif // _FLOAT_PRECISION_
}

#define TOT_DOF_X s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX
#define STRIDE 2 * ( (2*(TOT_DOF_X) -1)/2 +1 )

FFTW_freespace::FFTW_freespace(SimulationData& s) : PoissonSolver(s, STRIDE)
{
  assert(STRIDE/2==TOT_DOF_X); // Size of the domain must be an even number!
  printf("Employing FFTW-based Poisson solver by cyclic convolution.\n");
  const int desired_threads = omp_get_max_threads();
  #ifndef _FLOAT_PRECISION_
    const int retval = fftw_init_threads();
    fftw_plan_with_nthreads(desired_threads);
    buffer = fftw_alloc_real(2 * MY * MX_hat);
    fwd = fftw_plan_dft_r2c_2d(MY,MX, buffer,(mycomplex*)buffer, FFTW_MEASURE);
    bwd = fftw_plan_dft_c2r_2d(MY,MX, (mycomplex*)buffer,buffer, FFTW_MEASURE);
  #else // _FLOAT_PRECISION_
    const int retval = fftwf_init_threads();
    fftwf_plan_with_nthreads(desired_threads);
    buffer = fftwf_alloc_real(2 * MY * MX_hat);
    fwd = fftwf_plan_dft_r2c_2d(MY,MX, buffer,(mycomplex*)buffer, FFTW_MEASURE);
    bwd = fftwf_plan_dft_c2r_2d(MY,MX, (mycomplex*)buffer,buffer, FFTW_MEASURE);
  #endif // _FLOAT_PRECISION_
  if(retval==0) {
    cout << "Call fftw_init_threads() returned zero.\n"; fflush(0); abort();
  }

  {
    // This factor is due to the discretization of the convolution
    // integtal.  It is composed of (h*h) * (-1/[4*pi*h]), where h is the
    // uniform grid spacing.  The first factor is the discrete volume
    // element of the convolution integral; the second factor belongs to
    // Green's function on a uniform mesh. Multiplied both sides by h^2.

    //const Real h = sim.getH(), fac = h*h / ( 2*M_PI );
    const Real h = sim.getH(), fac = 1 / ( 2*M_PI );
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < MY; ++j)
    for (size_t i = 0; i < MX; ++i) {
        const Real yi = j >= totNy ? MY-j : j, xi = i >= totNx ? MX-i : i;
        buffer[i + 2*MX_hat*j] = fac * std::log(h * std::sqrt(xi*xi + yi*yi));
    }
    // Set self-interaction, which right now reads NaN (log(0))
    // G = 1/4 * r_eq^2 * (2* ln(r_eq) - 1) where  r_eq = h / sqrt(pi)
    buffer[0] = fac/2 * (2*std::log(h/std::sqrt(M_PI)) - 1);

    #ifndef _FLOAT_PRECISION_
     m_kernel = fftw_alloc_real(MY * MX_hat);
     fftw_execute(fwd);
    #else // _FLOAT_PRECISION_
     m_kernel = fftwf_alloc_real(MY * MX_hat);
     fftwf_execute(fwd);
    #endif // _FLOAT_PRECISION_

    const mycomplex *const G_hat = (mycomplex *) buffer;
    const Real norm_factor = 1.0 / (MY * MX);
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < MY; ++j)
    for (size_t i = 0; i < MX_hat; ++i)
      m_kernel[i +MX_hat*j] = G_hat[i +MX_hat*j][0] * norm_factor; // real part

    memset(buffer, 0, 2 * MY * MX_hat * sizeof(Real));
  }
}

#undef TOT_DOF_X
#undef STRIDE

#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <omp.h>

#include <fftw3.h>

#include "common.h"

#ifndef _FLOAT_PRECISION_
// double
typedef fftw_complex mycomplex;
typedef fftw_plan myplan;
#else // _FLOAT_PRECISION_
// float
typedef fftwf_complex mycomplex;
typedef fftwf_plan myplan;
#endif // _FLOAT_PRECISION_

using namespace std;

#include <BlockInfo.h>
#include <Profiler.h>

template<typename TGrid>
class PoissonSolverScalarFFTW
{
  typedef typename TGrid::BlockType BlockType;
  TGrid& grid;
  const vector<BlockInfo> infos = grid.getBlocksInfo();

  Profiler profiler;
  const int bs[2] = {BlockType::sizeX, BlockType::sizeY};
  // for fftw we swap x and y directions for ease of copying to/from grid
  const size_t nx = grid.getBlocksPerDimension(1)*bs[1];
  const size_t ny = grid.getBlocksPerDimension(0)*bs[0];
  const size_t mx = 2 * nx - 1;
  const size_t my = 2 * ny - 1;
  const size_t my_hat = my/2 +1;
  const Real h = grid.getBlocksInfo().front().h_gridpoint;

 protected:
  myplan fwd, bwd;
  Real * rhs; // rhs in _setup, out in cub2fftw and fftw2cub
  Real * m_kernel;

  void _setup()
  {
    #ifndef _FLOAT_PRECISION_
      const int retval = fftw_init_threads();
    #else // _FLOAT_PRECISION_
      const int retval = fftwf_init_threads();
    #endif // _FLOAT_PRECISION_
    if(retval==0) {
      cout << "FFTWBase::setup(): Oops the call to fftw_init_threads() returned zero. Aborting\n";
      abort();
    }

    const int desired_threads = omp_get_max_threads();

    #ifndef _FLOAT_PRECISION_
      fftw_plan_with_nthreads(desired_threads);
      rhs = fftw_alloc_real(2 * mx * my_hat);
      fwd = fftw_plan_dft_r2c_2d(mx, my, rhs, (mycomplex *)rhs, FFTW_MEASURE);
      bwd = fftw_plan_dft_c2r_2d(mx, my, (mycomplex *)rhs, rhs, FFTW_MEASURE);
    #else // _FLOAT_PRECISION_
      fftwf_plan_with_nthreads(desired_threads);
      rhs = fftwf_alloc_real(2 * mx * my_hat);
      fwd = fftwf_plan_dft_r2c_2d(mx, my, rhs, (mycomplex *)rhs, FFTW_MEASURE);
      bwd = fftwf_plan_dft_c2r_2d(mx, my, (mycomplex *)rhs, rhs, FFTW_MEASURE);
    #endif // _FLOAT_PRECISION_

    _init_green();
  }
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

  void _cub2fftw() const
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<infos.size(); ++i) {
      const BlockInfo& info = infos[i];
      BlockType& b = *(BlockType*)infos[i].ptrBlock;
      const size_t blocki = bs[0]*info.index[0], blockj = bs[1]*info.index[1];
      const size_t offset = blocki + 2*my_hat * blockj;

      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++) {
        const size_t dest_index = offset + ix + 2*my_hat * iy;
        rhs[dest_index] = b(ix,iy).tmp;
      }
    }
  }

  void _solve()
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

  void _fftw2cub() const
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<infos.size(); ++i) {
      const BlockInfo& info = infos[i];
      BlockType& b = *(BlockType*)infos[i].ptrBlock;
      const size_t blocki = bs[0]*info.index[0], blockj = bs[1]*info.index[1];
      const size_t offset = blocki + 2*my_hat * blockj;

      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++) {
        const size_t src_index = offset + ix + 2*my_hat * iy;
        b(ix,iy).tmp = rhs[src_index];
      }
    }
  }

 public:

  PoissonSolverScalarFFTW(TGrid& _grid) : grid(_grid) { _setup(); }

  void solve()
  {

    //profiler.push_start("CUB2FFTW");
    //_cub2fftw();
    //profiler.pop_stop();

    //profiler.push_start("FFTW FORWARD");
    #ifndef _FLOAT_PRECISION_
      fftw_execute(fwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(fwd);
    #endif // _FLOAT_PRECISION_
    //profiler.pop_stop();

    //profiler.push_start("SOLVE");
    _solve();
    //profiler.pop_stop();

    //profiler.push_start("FFTW INVERSE");
    #ifndef _FLOAT_PRECISION_
      fftw_execute(bwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(bwd);
    #endif // _FLOAT_PRECISION_
    //profiler.pop_stop();

    //profiler.push_start("FFTW2CUB");
    _fftw2cub();
    //profiler.pop_stop();

    //profiler.printSummary();
  }

  void dispose()
  {
    #ifndef _FLOAT_PRECISION_
    fftw_cleanup_threads();
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);

    fftw_free(rhs);
    fftw_free(m_kernel);

    #else // _FLOAT_PRECISION_
    fftwf_cleanup_threads();
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);

    fftwf_free(rhs);
    fftwf_free(m_kernel);

    #endif // _FLOAT_PRECISION_
  }

  inline size_t _offset(const BlockInfo &info) const {
    const size_t blocki = bs[0]*info.index[0], blockj = bs[1]*info.index[1];
    return blocki + 2*my_hat * blockj;
  }
  inline size_t _dest(const size_t offset, const int iy, const int ix) const {
    return offset + ix + 2*my_hat * iy;
  }
  inline void _cub2fftw(const size_t offset, const int iy, const int ix, const Real ret) const {
    const size_t dest_index = _dest(offset, iy, ix);
    rhs[dest_index] = ret;
  }
};

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
  const size_t nx = grid.getBlocksPerDimension(1)*bs[1];
  const size_t ny = grid.getBlocksPerDimension(0)*bs[0];
  const size_t ny_hat = ny/2 +1;
  const Real norm_factor = 1./(nx*ny);
  const Real h = grid.getBlocksInfo().front().h_gridpoint;

protected:
  myplan fwd, bwd;
  Real * rhs; // rhs in _setup, out in cub2fftw and fftw2cub

protected:

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
      rhs = fftw_alloc_real(2*nx*(ny/2+1)); // comes out of r2c/c2r (Section 2.3 of fftw3.pdf)

      fwd = fftw_plan_dft_r2c_2d(nx, ny, rhs, (mycomplex *)rhs, FFTW_MEASURE);
      bwd = fftw_plan_dft_c2r_2d(nx, ny, (mycomplex *)rhs, rhs, FFTW_MEASURE);
    #else // _FLOAT_PRECISION_
      fftwf_plan_with_nthreads(desired_threads);
      rhs = fftwf_alloc_real(2*nx*(ny/2+1)); // comes out of r2c/c2r (Section 2.3 of fftw3.pdf)

      fwd = fftwf_plan_dft_r2c_2d(nx, ny, rhs, (mycomplex *)rhs, FFTW_MEASURE);
      bwd = fftwf_plan_dft_c2r_2d(nx, ny, (mycomplex *)rhs, rhs, FFTW_MEASURE);
    #endif // _FLOAT_PRECISION_
  }

  void _cub2fftw()
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<infos.size(); ++i) {
      const BlockInfo& info = infos[i];
      BlockType& b = *(BlockType*)infos[i].ptrBlock;
      const size_t blocki = bs[0]*info.index[0], blockj = bs[1]*info.index[1];
      const size_t offset = blocki + 2*ny_hat * blockj;

      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++) {
        const size_t dest_index = offset + ix + 2*ny_hat * iy;
        rhs[dest_index] = b(ix,iy).tmp;
      }
    }
  }

  void _solve()
  {
    const Real h2 = h*h;
    const Real factor = h2*norm_factor;
    mycomplex * const in_out = (mycomplex *)rhs;

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<ny_hat; ++j)
      {
        const int linidx = i*ny_hat+j;

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
    in_out[0][0] = in_out[0][1] = 0;
  }

  void _solveSpectral()
  {
    mycomplex * const in_out = (mycomplex *)rhs;

    const Real waveFactX = 2.0*M_PI/(nx*h);
    const Real waveFactY = 2.0*M_PI/(ny*h);
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<ny_hat; ++j) {
        const int linidx = i * ny_hat + j;
        const int kx = (i <= nx/2) ? i : -(nx-i);
        const int ky = (j <= ny/2) ? j : -(ny-j);
        const Real rkx = kx*waveFactX;
        const Real rky = ky*waveFactY;
        const Real kinv = (kx==0 && ky==0) ? 0 : -1.0/(rkx*rkx+rky*rky);
        in_out[linidx][0] *= kinv*norm_factor;
        in_out[linidx][1] *= kinv*norm_factor;
      }

    //this is sparta!
    in_out[0][0] = in_out[0][1] = 0;
  }

  void _fftw2cub()
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<infos.size(); ++i) {
      const BlockInfo& info = infos[i];
      BlockType& b = *(BlockType*)infos[i].ptrBlock;
      const size_t blocki = bs[0]*info.index[0], blockj = bs[1]*info.index[1];
      const size_t offset = blocki + 2*ny_hat * blockj;

      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++) {
        const size_t src_index = offset + ix + 2*ny_hat * iy;
        b(ix,iy).tmp = rhs[src_index];
      }
    }
  }

public:

  PoissonSolverScalarFFTW(TGrid& _grid) : grid(_grid) { _setup(); }

  void solve(const bool spectral=true)
  {

    //profiler.push_start("CUB2FFTW");
    //  _cub2fftw();
    //profiler.pop_stop();

    profiler.push_start("FFTW FORWARD");
    #ifndef _FLOAT_PRECISION_
      fftw_execute(fwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(fwd);
    #endif // _FLOAT_PRECISION_
    profiler.pop_stop();

    profiler.push_start("SOLVE");
      if(spectral) _solveSpectral();
      else _solve();
    profiler.pop_stop();

    profiler.push_start("FFTW INVERSE");
    #ifndef _FLOAT_PRECISION_
      fftw_execute(bwd);
    #else // _FLOAT_PRECISION_
      fftwf_execute(bwd);
    #endif // _FLOAT_PRECISION_
    profiler.pop_stop();

    profiler.push_start("FFTW2CUB");
      _fftw2cub();
    profiler.pop_stop();

    //profiler.printSummary();
  }

  void dispose()
  {
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

  inline size_t _offset(const BlockInfo &info) const {
    const size_t blocki = bs[0]*info.index[0], blockj = bs[1]*info.index[1];
    return blocki + 2*ny_hat * blockj;
  }
  inline size_t _dest(const size_t offset, const int iy, const int ix) const {
    return offset + ix + 2*ny_hat * iy;
  }
  inline void _cub2fftw(const size_t offset, const int iy, const int ix, const Real ret) const {
    const size_t dest_index = _dest(offset, iy, ix);
    rhs[dest_index] = ret;
  }
};

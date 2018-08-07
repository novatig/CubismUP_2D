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

class PoissonSolverBase
{
 protected:
  typedef typename FluidGrid::BlockType BlockType;
  FluidGrid& grid;
  const vector<BlockInfo> infos = grid.getBlocksInfo();

  Profiler profiler;
  const int bs[2] = {BlockType::sizeX, BlockType::sizeY};
  const size_t nx = grid.getBlocksPerDimension(1)*bs[1];
  const size_t ny = grid.getBlocksPerDimension(0)*bs[0];
  const Real h = grid.getBlocksInfo().front().h_gridpoint;
  const size_t mx;
  const size_t my;
  const size_t my_hat = my/2 +1;

  myplan fwd, bwd;
  Real * rhs = nullptr; // rhs in _setup, out in cub2fftw and fftw2cub

  virtual void _solve() const = 0;

  inline void _cub2fftw() const {
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

  inline void _fftw2cub() const {
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

  PoissonSolverBase(FluidGrid& _grid, const bool bFrespace): grid(_grid),
    mx(bFrespace? 2 * nx - 1 : nx), my(bFrespace? 2 * ny - 1 : ny) {
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
  }

  void solve() const
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

  virtual ~PoissonSolverBase() {
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

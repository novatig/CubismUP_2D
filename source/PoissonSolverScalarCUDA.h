#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <omp.h>

#include <cufft.h>

#ifndef _FLOAT_PRECISION_
#define cufftCmpT cufftComplex
#else //_FLOAT_PRECISION_
#define cufftCmpT cufftDoubleComplex
#endif//_FLOAT_PRECISION_

#include <Grid.h>
#include <BlockInfo.h>
#include <BlockLab.h>

void freeCuMem(Real * buf);
void allocCuMem(Real* & ptr, const size_t size);
void freePlan(cufftHandle& plan);
void makePlan(cufftHandle& handle, const int mx, const int my, cufftType plan);
void cuSolve(const cufftHandle&fwd, const cufftHandle&bwd, const int mx,
  const int my, const Real h, Real*const rhs, Real*const rhs_gpu);

class PoissonSolverCuda
{
 protected:
  typedef typename FluidGrid::BlockType BlockType;
  FluidGrid& grid;
  const vector<BlockInfo> infos = grid.getBlocksInfo();

  Profiler profiler;
  const int bs[2] = {BlockType::sizeX, BlockType::sizeY};
  const int nx = grid.getBlocksPerDimension(1)*bs[1];
  const int ny = grid.getBlocksPerDimension(0)*bs[0];
  const Real h = grid.getBlocksInfo().front().h_gridpoint;
  const int mx, my, my_hat = my/2 +1;
  const Real facX = 2.0*M_PI/(mx*h), facY = 2.0*M_PI/(my*h), norm = 1./(mx*my);

  cufftHandle fwd, bwd;
  Real * rhs = nullptr;
  Real * rhs_gpu = nullptr;

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

  PoissonSolverCuda(FluidGrid*const _grid) :
  grid(*_grid), mx(nx), my(ny)
  {
    makePlan(fwd, mx, my, CUFFT_R2C);
    makePlan(bwd, mx, my, CUFFT_C2R);
    assert(2*sizeof(Real) == sizeof(ccufftCmpT));
    rhs = (Real*) malloc(mx * my_hat * 2 * sizeof(Real) );
    allocCuMem(rhs_gpu, mx * my_hat * sizeof(cufftCmpT) );
  }

  void solve() const
  {
    cuSolve(fwd, bwd, mx, my, h, rhs, rhs_gpu);
    _fftw2cub();
  }

  virtual ~PoissonSolverCuda() {
    freePlan(fwd);
    freePlan(bwd);
    freeCuMem(rhs_gpu);
    free(rhs);
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

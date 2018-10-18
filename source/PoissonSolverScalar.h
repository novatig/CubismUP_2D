#pragma once

#include <omp.h>
#include "common.h"

class PoissonSolverBase
{
 protected:
  typedef typename FluidGrid::BlockType BlockType;
  FluidGrid& grid;
  const vector<BlockInfo> infos = grid.getBlocksInfo();

  const int bs[2] = {BlockType::sizeX, BlockType::sizeY};
  const size_t nx = grid.getBlocksPerDimension(1)*bs[1];
  const size_t ny = grid.getBlocksPerDimension(0)*bs[0];
  const Real h = grid.getBlocksInfo().front().h_gridpoint;
  const size_t mx;
  const size_t my;
  const size_t my_hat;

  Real * rhs = nullptr; // rhs in _setup, out in cub2fftw and fftw2cub

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

  PoissonSolverBase(FluidGrid& _grid, const bool bFreeSpace, long _my_hat= -1) :
  grid(_grid), mx(bFreeSpace? 2*nx-1 : nx), my(bFreeSpace? 2*ny-1 : ny),
  my_hat(_my_hat<1 ? my/2 +1 : _my_hat) { }

  virtual void solve() const = 0;

  virtual ~PoissonSolverBase() { }

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

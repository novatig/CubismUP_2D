
#include "PoissonSolverScalar.h"


void PoissonSolverBase::_cub2fftw() const
{
  const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const size_t nBlocks = tmpInfo.size();
  Real sumRHS = 0, sumPos = 0, sumNeg = 0;

  Real * __restrict__ const dest = rhs;

  #pragma omp parallel for schedule(static) reduction(+:sumRHS,sumPos,sumNeg)
  for(size_t i=0; i<nBlocks; ++i)
  {
    const BlockInfo& info = tmpInfo[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    const ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    const size_t blockStart = blocki + 2*my_hat * blockj;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++) {
      dest[blockStart + ix + 2*my_hat*iy] = b(ix,iy).s;
      sumPos += (b(ix,iy).s > 0) * b(ix,iy).s;
      sumNeg += (b(ix,iy).s < 0) * b(ix,iy).s;
      sumRHS +=  b(ix,iy).s;
    }
  }

  if(sumRHS>0)
  {
    const Real correction = sumRHS / sumPos;
    printf("Relative RHS correction:%f\n", correction);
    #pragma omp parallel for schedule(static)
    for (size_t iy = 0; iy < nx; iy++)
    for (size_t ix = 0; ix < ny; ix++)
      if ( dest[ix + 2*my_hat*iy] > 0 )
        dest[ix + 2*my_hat * iy] -=  dest[ix + 2*my_hat * iy] * correction;
  }
  else
  {
    const Real correction = sumRHS / sumNeg;
    printf("Relative RHS correction:%f\n", correction);
    #pragma omp parallel for schedule(static) 
    for (size_t iy = 0; iy < nx; iy++)
    for (size_t ix = 0; ix < ny; ix++)
      if ( dest[ix + 2*my_hat*iy] < 0 )
        dest[ix + 2*my_hat * iy] -=  dest[ix + 2*my_hat * iy] * correction;
  }
}

void PoissonSolverBase::_fftw2cub() const
{
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const size_t nBlocks = presInfo.size();

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<nBlocks; ++i)
  {
    const BlockInfo& info = presInfo[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    const size_t blockStart = blocki + 2*my_hat * blockj;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
      b(ix,iy).s = rhs[blockStart + ix + 2*my_hat*iy];
  }
}

PoissonSolverBase::PoissonSolverBase(SimulationData& s,
  const bool bFreeSpace, long _my_hat) : sim(s), mx(bFreeSpace? 2*nx-1 : nx), my(bFreeSpace? 2*ny-1 : ny), my_hat(_my_hat<1 ? my/2 +1 : _my_hat)
{ }

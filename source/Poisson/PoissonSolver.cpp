
#include "PoissonSolver.h"


void PoissonSolver::cub2rhs()
{
  const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const size_t nBlocks = tmpInfo.size();
  Real sumRHS = 0, sumABS = 0;

  Real * __restrict__ const dest = buffer;

  #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
  for(size_t i=0; i<nBlocks; ++i)
  {
    const BlockInfo& info = tmpInfo[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    const ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    const size_t blockStart = blocki + stride * blockj;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++) {
      dest[blockStart + ix + stride*iy] = b(ix,iy).s;
      sumABS += std::fabs(b(ix,iy).s);
      sumRHS +=           b(ix,iy).s;
    }
  }

  const Real correction = sumRHS / sumABS;
  printf("Relative RHS correction:%e\n", correction);
  #pragma omp parallel for schedule(static)
  for (size_t iy = 0; iy < totNy; iy++)
  for (size_t ix = 0; ix < totNx; ix++)
    dest[ix + stride * iy] -=  std::fabs(dest[ix +stride * iy]) * correction;

  #ifndef NDEBUG
    Real sumRHSpost = 0;
    #pragma omp parallel for schedule(static) reduction(+ : sumRHSpost)
    for (size_t iy = 0; iy < totNy; iy++)
    for (size_t ix = 0; ix < totNx; ix++) sumRHSpost += dest[ix + stride * iy];
    printf("Relative RHS correction:%e\n", sumRHSpost);
    assert(sumRHSpost < std::sqrt(std::numeric_limits<Real>::epsilon()));
  #endif
}

void PoissonSolver::sol2cub()
{
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const size_t nBlocks = presInfo.size();

  Real _avgP = 0;
  const Real fac = 1.0 / (totNx * totNy);
  const Real * __restrict__ const sorc = buffer;
  #pragma omp parallel for schedule(static) reduction(+:_avgP)
  for(size_t i=0; i<nBlocks; ++i)
  {
    const BlockInfo& info = presInfo[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    const size_t blockStart = blocki + stride * blockj;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++) {
      const Real P = sorc[blockStart + ix + stride * iy];
      _avgP += fac * P; b(ix,iy).s = P;
    }
  }
  avgP = _avgP;
}

PoissonSolver::PoissonSolver(SimulationData&s,long p): sim(s), stride(p) {}

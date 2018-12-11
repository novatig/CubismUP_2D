#pragma once

#include "../Operator.h"

class PoissonSolverBase
{
 protected:
  SimulationData& sim;

  const size_t nx = sim.vel->getBlocksPerDimension(1)*VectorBlock::sizeY;
  const size_t ny = sim.vel->getBlocksPerDimension(0)*VectorBlock::sizeX;
  const size_t mx;
  const size_t my;
  const size_t my_hat;

  Real * rhs = nullptr; // rhs in _setup, out in cub2fftw and fftw2cub

  void _cub2fftw() const;

  void _fftw2cub() const;

 public:

  PoissonSolverBase(SimulationData& s, const bool bFreeSpace, long _my_hat=-1);

  virtual void solve() const = 0;

  virtual ~PoissonSolverBase() { }
};

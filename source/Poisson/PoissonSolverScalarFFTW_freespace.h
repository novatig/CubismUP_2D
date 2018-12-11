#pragma once

#include "PoissonSolverScalar.h"

#include <fftw3.h>
#ifndef _FLOAT_PRECISION_
typedef fftw_complex mycomplex;
typedef fftw_plan myplan;
#else // _FLOAT_PRECISION_
typedef fftwf_complex mycomplex;
typedef fftwf_plan myplan;
#endif // _FLOAT_PRECISION_

class PoissonSolverFreespace: public PoissonSolverBase
{
  Real * m_kernel = nullptr;

  myplan fwd, bwd;
 protected:

  void _init_green();

 public:
  PoissonSolverFreespace(SimulationData& s);

  void solve() const override;

  ~PoissonSolverFreespace();
};

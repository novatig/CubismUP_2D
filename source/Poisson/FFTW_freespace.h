//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "PoissonSolver.h"

#include <fftw3.h>
#ifndef _FLOAT_PRECISION_
typedef fftw_complex mycomplex;
typedef fftw_plan myplan;
#else // _FLOAT_PRECISION_
typedef fftwf_complex mycomplex;
typedef fftwf_plan myplan;
#endif // _FLOAT_PRECISION_

class FFTW_freespace: public PoissonSolver
{
  Real * m_kernel = nullptr;
  const size_t MX = 2*totNx - 1;
  const size_t MY = 2*totNy - 1;
  const size_t MX_hat = MX/2 +1;
  myplan fwd, bwd;

 public:
  FFTW_freespace(SimulationData& s);

  void solve(const std::vector<BlockInfo>& BSRC,
             const std::vector<BlockInfo>& BDST) override;

  ~FFTW_freespace();
};

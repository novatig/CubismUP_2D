#pragma once

#include <PoissonSolverScalarFFTW.h>

class PoissonSolverPeriodic : public PoissonSolverBase
{
  const Real norm_factor = 1./(nx*ny);

 protected:

  void _solve_finiteDiff()
  {
    const Real h2 = h*h;
    const Real factor = h2*norm_factor;
    mycomplex * const in_out = (mycomplex *)rhs;

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<my_hat; ++j)
      {
        const int linidx = i*my_hat+j;

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

  void _solve() const override
  {
    mycomplex * const in_out = (mycomplex *)rhs;

    const Real waveFactX = 2.0*M_PI/(nx*h);
    const Real waveFactY = 2.0*M_PI/(ny*h);
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nx; ++i)
      for(size_t j = 0; j<my_hat; ++j) {
        const int linidx = i * my_hat + j;
        const int kx = (i <= nx/2) ? i : -(nx-i);
        const int ky = (j <= ny/2) ? j : -(ny-j);
        const Real rkx = kx*waveFactX;
        const Real rky = ky*waveFactY;
        const Real kinv = (kx==0 && ky==0) ? 0 : -1/(rkx*rkx+rky*rky);
        in_out[linidx][0] *= kinv*norm_factor;
        in_out[linidx][1] *= kinv*norm_factor;
      }

    //this is sparta!
    in_out[0][0] = in_out[0][1] = 0;
  }

public:

  PoissonSolverPeriodic(FluidGrid*const _grid) :
    PoissonSolverBase(*_grid, false) { }
};

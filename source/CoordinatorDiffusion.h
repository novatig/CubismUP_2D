//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorDiffusion_h
#define CubismUP_2D_CoordinatorDiffusion_h

#include "GenericCoordinator.h"

class OperatorViscousDrag : public GenericLabOperator
{
 private:
	double dt;
	Real viscousDrag;

 public:
	OperatorViscousDrag(double dt) : dt(dt), viscousDrag(0)
	{
		stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
		stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
	}

	~OperatorViscousDrag() {}

	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
	{
		const double prefactor = 1. / (info.h_gridpoint*info.h_gridpoint);
		viscousDrag = 0;
		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
		for(int ix=0; ix<FluidBlock::sizeX; ++ix)
		{
			FluidElement& phi = lab(ix,iy);
			FluidElement& phiN = lab(ix,iy+1);
			FluidElement& phiS = lab(ix,iy-1);
			FluidElement& phiE = lab(ix+1,iy);
			FluidElement& phiW = lab(ix-1,iy);
			viscousDrag += prefactor * (phiN.tmp + phiS.tmp + phiE.tmp + phiW.tmp - 4.*phi.tmp);
		}
	}

	inline Real getDrag()
	{
		return viscousDrag;
	}
};

class OperatorDiffusion : public GenericLabOperator
{
 private:
	const double mu, dt;

 public:
	OperatorDiffusion(double dt, double mu) : mu(mu), dt(dt)
	{
		stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
		stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
	}

	~OperatorDiffusion() {}

	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
	{
		const double prefactor = mu * dt / (info.h_gridpoint*info.h_gridpoint);

		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
		for(int ix=0; ix<FluidBlock::sizeX; ++ix)
		{
			FluidElement& phi  = lab(ix,iy);
			FluidElement& phiN = lab(ix,iy+1);
			FluidElement& phiS = lab(ix,iy-1);
			FluidElement& phiE = lab(ix+1,iy);
			FluidElement& phiW = lab(ix-1,iy);
			const Real fac = prefactor/phi.rho;
			o(ix,iy).tmpU = phi.u + fac * (phiN.u + phiS.u + phiE.u + phiW.u - phi.u*4.);
			o(ix,iy).tmpV = phi.v + fac * (phiN.v + phiS.v + phiE.v + phiW.v - phi.v*4.);
      #ifdef _MULTIPHASE_
			o(ix,iy).tmp = phi.rho + fac *(phiN.rho + phiS.rho + phiE.rho + phiW.rho - phi.rho*4.);
      #endif
		}
   }
};

template <typename Lab>
class CoordinatorDiffusion : public GenericCoordinator
{
 protected:
  const double coeff;
  Real *uBody, *vBody;
  Real *viscousDrag;

	inline void reset()
	{
		const int N = vInfo.size();
    #pragma omp parallel for schedule(static)
		for(int i=0; i<N; i++)
		{
			BlockInfo info = vInfo[i];
			FluidBlock& b = *(FluidBlock*)info.ptrBlock;

			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
				for(int ix=0; ix<FluidBlock::sizeX; ++ix)
				{
					b(ix,iy).tmpU = 0;
					b(ix,iy).tmpV = 0;
          #ifdef _MULTIPHASE_
					b(ix,iy).tmp = 0;
          #endif
				}
		}
	};

	inline void update()
	{
		const int N = vInfo.size();
    #pragma omp parallel for schedule(static)
		for(int i=0; i<N; i++)
		{
			BlockInfo info = vInfo[i];
			FluidBlock& b = *(FluidBlock*)info.ptrBlock;

			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
				for(int ix=0; ix<FluidBlock::sizeX; ++ix)
				{
					b(ix,iy).u = b(ix,iy).tmpU;
					b(ix,iy).v = b(ix,iy).tmpV;
          #ifdef _MULTIPHASE_
					b(ix,iy).rho = b(ix,iy).tmp;
          #endif
				}
		}
	 }

	inline void diffuse(const double dt, const int stage)
	{
		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();

    #pragma omp parallel
		{
			OperatorDiffusion kernel(dt, coeff);

      Lab mylab;
      #ifdef _MOVING_FRAME_
      mylab.pDirichlet.u = *uBody;
      mylab.pDirichlet.v = *vBody;
      #endif
			mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(static)
			for (int i=0; i<N; i++)
			{
				mylab.load(ary[i], 0);
				kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
			}
		}
	}

  inline void drag()
  {
    BlockInfo * ary = &vInfo.front();
    const int N = vInfo.size();
    *viscousDrag = 0;
    Real tmpDrag = 0;

    #pragma omp parallel
    {
      OperatorViscousDrag kernel(0);

      Lab mylab;
      #ifdef _MOVING_FRAME_
      mylab.pDirichlet.u = *uBody;
      mylab.pDirichlet.v = *vBody;
      #endif
      mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(static) reduction(+:tmpDrag)
      for (int i=0; i<N; i++)
      {
        mylab.load(ary[i], 0);
        kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
        tmpDrag += kernel.getDrag();
      }
    }

		*viscousDrag = tmpDrag*coeff;
  }

 public:
	CoordinatorDiffusion(const double coeff, Real * uBody, Real * vBody, Real *viscousDrag, FluidGrid * grid) :
  GenericCoordinator(grid), coeff(coeff), uBody(uBody), vBody(vBody), viscousDrag(viscousDrag)
	{
	}

	void operator()(const double dt)
	{
		check("diffusion - start");

		reset();
		diffuse(dt,0);
		update();
    drag();

		check("diffusion - end");
	}

	string getName()
	{
		return "Diffusion";
	}
};

#endif

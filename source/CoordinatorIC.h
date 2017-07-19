//
//  CoordinatorIC.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorIC_h
#define CubismUP_2D_CoordinatorIC_h

#include "GenericCoordinator.h"
#include "Shape.h"

class OperatorIC : public GenericOperator
{
 private:
	Shape * shape;
	const double uinfx, uinfy;

 public:
	OperatorIC(Shape * shape, const double uinfx, const double uinfy) :
	shape(shape), uinfx(uinfx), uinfy(uinfy) {}

	~OperatorIC() {}

	void operator()(const BlockInfo& info, FluidBlock& block) const
	{
		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
			for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				Real p[2];
				info.pos(p, ix, iy);

				block(ix,iy).u = uinfx;
				block(ix,iy).v = uinfy;
				block(ix,iy).rho = 1;

				block(ix,iy).p = 0;
				block(ix,iy).pOld = 0;

				block(ix,iy).tmpU = 0;
				block(ix,iy).tmpV = 0;
				block(ix,iy).tmp  = 0;
			}
	}
};

class CoordinatorIC : public GenericCoordinator
{
 protected:
	Shape * shape;
	const double uinfx, uinfy;

	public:
	CoordinatorIC(Shape * shape, const double uinfx, const double uinfy, FluidGrid * grid) :
	 GenericCoordinator(grid), shape(shape), uinfx(uinfx), uinfy(uinfy)
	{
	}

	void operator()(const double dt)
	{
		const int N = vInfo.size();

		#pragma omp parallel
		{
			OperatorIC kernel(shape, uinfx, uinfy);

			#pragma omp for schedule(static)
			for (int i=0; i<N; i++)
				kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
		}

		check("IC - end");
	}

	string getName()
	{
		return "IC";
	}
};

class OperatorFadeOut : public GenericOperator
{
 private:
	const Real extent[2], uinfx, uinfy;
	const int info[2], buffer;

   inline bool _is_touching(const BlockInfo& i, const Real h) const
	{
		Real min_pos[2], max_pos[2];
    i.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
		i.pos(min_pos, 0, 0);
    const bool touchN = max_pos[1] > extent[1]-(2+buffer)*h;
    const bool touchE = max_pos[0] > extent[0]-(2+buffer)*h;
    const bool touchS = min_pos[1] < 0 +(2+buffer)*h;
    const bool touchW = min_pos[0] < 0 +(2+buffer)*h;
    return touchN || touchE || touchS || touchW;
	}

 public:
	OperatorFadeOut(const int info[2], const int buffer, const Real extent[2], const Real uinfx, const Real uinfy)
	: extent{extent[0],extent[1]}, uinfx(uinfx), uinfy(uinfy), info{info[0],info[1]}, buffer(buffer) {}

	void operator()(const BlockInfo& i, FluidBlock& b) const
	{
		const Real h = i.h_gridpoint;
		const Real iWidth = 1/(buffer*h);
		const int ax = info[0];
		const int dir = info[1];

		if(_is_touching(i,h))
		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
		for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
			Real p[2];
			i.pos(p, ix, iy);
         const Real arg1= max(0.,  p[0] -extent[0] +(2+buffer)*h );
         const Real arg2= max(0.,  p[1] -extent[1] +(2+buffer)*h );
         const Real arg3= max(0., -p[0] +(2+buffer)*h );
         const Real arg4= max(0., -p[1] +(2+buffer)*h );
			const Real dist= max(max(arg1, arg2), max(arg3, arg4));
			const Real fade= max(0., cos(.5*M_PI*dist*iWidth) );
			b(ix,iy).u = b(ix,iy).u*fade + (1-fade)*uinfx;
			b(ix,iy).v = b(ix,iy).v*fade + (1-fade)*uinfy;
		}
	}
};

class CoordinatorFadeOut : public GenericCoordinator
{
 protected:
	const int buffer;
	const Real uinfx, uinfy;
	const Real * const uBody;
	const Real * const vBody;

 public:
  CoordinatorFadeOut(Real* uBody, Real* vBody, Real uinfx, Real uinfy, FluidGrid* grid, int _buffer=8)
	: GenericCoordinator(grid), buffer(_buffer), uBody(uBody), vBody(vBody), uinfx(uinfx), uinfy(uinfy)
	{ }

	void operator()(const Real dt)
	{
		check((string)"FadeOut - start");

		const int N = vInfo.size();
		const Real ext[2] = {
      vInfo[0].h_gridpoint * grid->getBlocksPerDimension(0) * FluidBlock::sizeX,
      vInfo[0].h_gridpoint * grid->getBlocksPerDimension(1) * FluidBlock::sizeY
		};
		const int movey = fabs(uinfy-*vBody) > fabs(uinfx-*uBody);
		const int dirs[2] = {*uBody-uinfx>0 ? 1 : -1, *vBody-uinfy>0 ? 1 : -1};
		const int info[2] = {movey, dirs[movey]};

		#pragma omp parallel
		{
			OperatorFadeOut kernel(info, buffer, ext, uinfx, uinfy);
			#pragma omp for schedule(static)
			for (int i=0; i<N; i++)
				kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
		}

		check((string)"FadeOut - end");
	}

	string getName()
	{
		return "FadeOut";
	}
};

#endif

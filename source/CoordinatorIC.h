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
#include "OperatorIC.h"
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
				block(ix,iy).chi = shape->chi(p, info.h_gridpoint);

				// assume fluid with density 1
				block(ix,iy).rho = shape->rho(p, info.h_gridpoint);

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
		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();

		#pragma omp parallel
		{
			OperatorIC kernel(shape, uinf);

			#pragma omp for schedule(static)
			for (int i=0; i<N; i++)
				kernel(ary[i], *(FluidBlock*)ary[i].ptrBlock);
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
	const int info[2];
	const Real extent[2];
	const int buffer;

   inline bool _is_touching(const BlockInfo& i, const Real h) const
	{
		Real max_pos[3],min_pos[3];
		const int ax = info[0];
		const int dir = info[1];
		if(dir>0) //moving up
		{
			i.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1, FluidBlock::sizeZ-1);
			return max_pos[ax] > extent[ax]-(2+buffer)*h;
		}
		else //moving down
		{
			i.pos(min_pos, 0, 0, 0);
			return min_pos[ax] < 0 +(2+buffer)*h;
		}
	}

public:
	OperatorFadeOut(const int info[2], const int buffer, const Real extent[2])
	: info{info[0],info[1]}, extent{extent[0],extent[1]}, buffer(buffer) {}

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
			const Real dist = dir>0 ? p[ax]-extent[ax]+(2+buffer)*h
				                : 0.0  -p[0]      +(2+buffer)*h;
			const Real fade = max(Real(0), cos(.5*M_PI*max(0.,dist)*iWidth) );
			b(ix,iy).u = b(ix,iy).u*fade;
			b(ix,iy).v = b(ix,iy).v*fade;
		}
	}
};

class CoordinatorFadeOut : public GenericCoordinator
{
 protected:
	const int buffer;
	const Real *uBody, *vBody;

public:
    CoordinatorFadeOut(Real * uBody, Real * vBody, FluidGrid * grid, const int _buffer=8)
	: GenericCoordinator(grid), buffer(_buffer), uBody(uBody), vBody(vBody)
	{ }

	void operator()(const Real dt)
	{
		check((string)"FadeOut - start");

		const int N = vInfo.size();
		const Real ext[2] = {
				vInfo[0].h_gridpoint * grid->getBlocksPerDimension(0) * FluidBlock::sizeX,
				vInfo[0].h_gridpoint * grid->getBlocksPerDimension(1) * FluidBlock::sizeY
		};
		const int movey = fabs(*vBody) > fabs(*uBody);
		const int dirs[2] = {*uBody>0 ? 1 : -1, *vBody>0 ? 1 : -1};
		const int info[2] = {movey, dirs[movey]};

		#pragma omp parallel
		{
			OperatorFadeOut kernel(info, buffer, ext);
			#pragma omp for schedule(static)
			for (int i=0; i<N; i++) {
				FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
				kernel(vInfo[i], b);
			}
		}

		check((string)"FadeOut - end");
	}

	string getName()
	{
		return "FadeOut";
	}
};

#endif

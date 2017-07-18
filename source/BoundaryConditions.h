/*
 *  BoundaryConditions.h
 *  MPCFnode
 *
 *  Created by Babak Hejazialhosseini  on 6/16/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "BlockLab.h"
#include <Matrix3D.h>

template<typename TBlock, typename TElement, template<typename X> class allocator=std::allocator>
class BoundaryCondition
{
 protected:

	int s[3], e[3];
	int stencilStart[3], stencilEnd[3];
	Matrix3D<TElement, true, allocator> * cacheBlock;

	// this guy includes corners!
	// what happens if they are not allocated?
	template<int dir, int side>
	void _setup()
	{
		s[0] =	dir==0 ? (side==0 ? stencilStart[0]: TBlock::sizeX) : stencilStart[0];
		s[1] =	dir==1 ? (side==0 ? stencilStart[1]: TBlock::sizeY) : stencilStart[1];
		s[2] =	dir==2 ? (side==0 ? stencilStart[2]: TBlock::sizeZ) : stencilStart[2];

		e[0] =	dir==0 ? (side==0 ? 0: TBlock::sizeX + stencilEnd[0]-1) : TBlock::sizeX +  stencilEnd[0]-1;
		e[1] =	dir==1 ? (side==0 ? 0: TBlock::sizeY + stencilEnd[1]-1) : TBlock::sizeY +  stencilEnd[1]-1;
		e[2] =	dir==2 ? (side==0 ? 0: TBlock::sizeZ + stencilEnd[2]-1) : TBlock::sizeZ +  stencilEnd[2]-1;
	}
 public:

	BoundaryCondition(const int ss[3], const int se[3], Matrix3D<TElement, true, allocator> * cacheBlock):
	cacheBlock(cacheBlock)
	{
		s[0]=s[1]=s[2]=0;
		e[0]=e[1]=e[2]=0;

		stencilStart[0] = ss[0];
		stencilStart[1] = ss[1];
		stencilStart[2] = ss[2];

		stencilEnd[0] = se[0];
		stencilEnd[1] = se[1];
		stencilEnd[2] = se[2];
	}

	TElement& operator()(int ix, int iy)
	{
		return cacheBlock->Access(ix-stencilStart[0],iy-stencilStart[1], 0);
	}

	template<int dir, int side>
	void applyBC_absorbing()
	{
		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
		for(int ix=s[0]; ix<e[0]; ix++)
		{
			(*this)(ix,iy) = (*this)(dir==0? (side==0? 0:TBlock::sizeX-1):ix,
										dir==1? (side==0? 0:TBlock::sizeY-1):iy);
		}
	}

	template<int dir, int side>
	void applyBC_mixedBottom(const TElement& p)
	{
		assert(dir==1);
		assert(side==0);

		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
		for(int ix=s[0]; ix<e[0]; ix++)
		{
			(*this)(ix,iy).rho  = p.rho;
			(*this)(ix,iy).tmp  = p.tmp;

			// dirichlet BC
			(*this)(ix,iy).u = 2*p.u - (*this)(ix, -iy-1).u;
			(*this)(ix,iy).v = 2*p.v - (*this)(ix, -iy-1).v;
			(*this)(ix,iy).tmpU = 2*p.tmpU - (*this)(ix, -iy-1).tmpU;
			(*this)(ix,iy).tmpV = 2*p.tmpV - (*this)(ix, -iy-1).tmpV;

			// Neumann BC
			(*this)(ix,iy).p    = (*this)(ix, -iy-1).p;
			(*this)(ix,iy).pOld = (*this)(ix, -iy-1).pOld;
			//(*this)(ix,iy).divU = (*this)(ix, -iy-1).divU;
		}
	}

	template<int dir, int side>
	void applyBC_mixedTop(const TElement& p)
	{
		assert(dir==1);
        assert(side==1);

		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
			for(int ix=s[0]; ix<e[0]; ix++)
            {
                (*this)(ix,iy).rho  = (*this)(ix, TBlock::sizeY-1).rho;//p.rho;
                (*this)(ix,iy).tmp  = (*this)(ix, TBlock::sizeY-1).tmp;//p.rho;

                // dirichlet BC
                (*this)(ix,iy).p    = 2*p.p    - (*this)(ix,2*TBlock::sizeY-1-iy).p;
                (*this)(ix,iy).pOld = 2*p.pOld - (*this)(ix,2*TBlock::sizeY-1-iy).pOld;
                //(*this)(ix,iy).divU = 2*p.divU - (*this)(ix,2*TBlock::sizeY-1-iy).divU; // needed because we compute gradP on this!

                // Neumann BC
                (*this)(ix,iy).u = (*this)(ix, 2*TBlock::sizeY-1-iy).u;
                (*this)(ix,iy).v = (*this)(ix, 2*TBlock::sizeY-1-iy).v;
                (*this)(ix,iy).tmpU = (*this)(ix, 2*TBlock::sizeY-1-iy).tmpU;
                (*this)(ix,iy).tmpV = (*this)(ix, 2*TBlock::sizeY-1-iy).tmpV;
			}
	}

	template<int dir, int side>
	void applyBC_BoxLeft(const TElement& p)
	{
		assert(dir==0);
		assert(side==0);

		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
		for(int ix=s[0]; ix<e[0]; ix++)
		{
			(*this)(ix,iy).rho  = p.rho;
			(*this)(ix,iy).tmp  = p.tmp;

			// dirichlet BC
			(*this)(ix,iy).u = 2*p.u - (*this)(-ix-1, iy).u;
			(*this)(ix,iy).v = 2*p.v - (*this)(-ix-1, iy).v;
			(*this)(ix,iy).tmpU = 2*p.tmpU - (*this)(-ix-1, iy).tmpU;
			(*this)(ix,iy).tmpV = 2*p.tmpV - (*this)(-ix-1, iy).tmpV;

			// Neumann BC
			(*this)(ix,iy).p    = (*this)(-ix-1,iy).p;
			(*this)(ix,iy).pOld = (*this)(-ix-1,iy).pOld;
			//(*this)(ix,iy).divU = (*this)(-ix-1,iy).divU;
		}
	}

	template<int dir, int side>
	void applyBC_BoxRight(const TElement& p)
	{
		assert(dir==0);
		assert(side==1);

		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
		for(int ix=s[0]; ix<e[0]; ix++)
		{
			(*this)(ix,iy).rho  = p.rho;
			(*this)(ix,iy).tmp  = p.tmp;

			// dirichlet BC
			(*this)(ix,iy).u = 2*p.u - (*this)(2*TBlock::sizeX-1-ix,iy).u;
			(*this)(ix,iy).v = 2*p.v - (*this)(2*TBlock::sizeX-1-ix,iy).v;
			(*this)(ix,iy).tmpU = 2*p.tmpU - (*this)(2*TBlock::sizeX-1-ix,iy).tmpU;
			(*this)(ix,iy).tmpV = 2*p.tmpV - (*this)(2*TBlock::sizeX-1-ix,iy).tmpV;

			// Neumann BC
			(*this)(ix,iy).p    = (*this)(2*TBlock::sizeX-1-ix,iy).p;
			(*this)(ix,iy).pOld = (*this)(2*TBlock::sizeX-1-ix,iy).pOld;
			//(*this)(ix,iy).divU = (*this)(2*TBlock::sizeX-1-ix,iy).divU;
		}
	}

	template<int dir, int side>
	void applyBC_BoxTop(const TElement& p)
	{
		assert(dir==1);
		assert(side==1);

		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
		for(int ix=s[0]; ix<e[0]; ix++)
		{
			(*this)(ix,iy).rho  = p.rho;
			(*this)(ix,iy).tmp  = p.tmp;

			// dirichlet BC
			(*this)(ix,iy).u = 2*p.u - (*this)(ix,2*TBlock::sizeY-1-iy).u;
			(*this)(ix,iy).v = 2*p.v - (*this)(ix,2*TBlock::sizeY-1-iy).v;
			(*this)(ix,iy).tmpU = 2*p.tmpU - (*this)(ix,2*TBlock::sizeY-1-iy).tmpU;
			(*this)(ix,iy).tmpV = 2*p.tmpV - (*this)(ix,2*TBlock::sizeY-1-iy).tmpV;

			// Neumann BC
			(*this)(ix,iy).p    = (*this)(ix,2*TBlock::sizeY-1-iy).p;
			(*this)(ix,iy).pOld = (*this)(ix,2*TBlock::sizeY-1-iy).pOld;
			//(*this)(ix,iy).divU = (*this)(ix,2*TBlock::sizeY-1-iy).divU;
		}
	}

	template<int dir, int side>
	void applyBC_vortex(const BlockInfo info)
	{
		_setup<dir,side>();

		for(int iy=s[1]; iy<e[1]; iy++)
			for(int ix=s[0]; ix<e[0]; ix++)
			{
				/*
				(*this)(ix,iy).rho = 1;
				(*this)(ix,iy).u = 0;
				(*this)(ix,iy).v = 0;
				(*this)(ix,iy).p = 0;
				*/
				 //*
				double p[3];
				info.pos(p, ix, iy);

				p[0] = p[0]*2.-1.;
				p[1] = p[1]*2.-1.;

                const Real r = sqrt(p[0]*p[0] + p[1]*p[1]);
				const Real invR = 1./r;

        (*this)(ix,iy).rho = r;
        (*this)(ix,iy).u   =   sin(p[1])*cos(r*M_PI/2)*invR;//-p[1];//
        (*this)(ix,iy).v   =  -sin(p[0])*cos(r*M_PI/2)*invR;// p[0];//
				 // what about pressure?
				 //*/
			}
	}
};

//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_CoordinatorPressure_h
#define CubismUP_2D_CoordinatorPressure_h

#include "GenericCoordinator.h"
#include "OperatorDivergence.h"
#include "OperatorGradP.h"
#include "PoissonSolverScalarFFTW.h"

//#define _HYDROSTATIC_

template <typename Lab>
class CoordinatorPressure : public GenericCoordinator
{
protected:
	const int rank, nprocs;
	const double minRho;
	Real gravity[2];
	int * step;
    const bool bSplit;
    Real *uBody, *vBody;
    Real *pressureDragX, *pressureDragY;
	
#ifndef _MIXED_
#ifndef _BOX_
#ifndef _OPENBOX_
	PoissonSolverScalarFFTW<FluidGrid, StreamerDiv> pressureSolver;
#else
	PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
#endif // _OPENBOX_
#else
	PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
#endif // _BOX_
#else
	PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
#endif // _MIXED_
	
	inline void addHydrostaticPressure(const double dt)
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
					b(ix,iy).u -= dt*gravity[0]/b(ix,iy).rho;
					b(ix,iy).v -= dt*gravity[1]/b(ix,iy).rho;
					
					// doesn't help much
					//b(ix,iy).u -= dt*minRho*gravity[0]/b(ix,iy).rho + (minRho<1 ? dt*(1-minRho)*gravity[0] : 0);
					//b(ix,iy).v -= dt*minRho*gravity[1]/b(ix,iy).rho + (minRho<1 ? dt*(1-minRho)*gravity[1] : 0);
				}
		}
	}
	
	inline void updatePressure()
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
					b(ix,iy).pOld = b(ix,iy).p;
					b(ix,iy).p    = b(ix,iy).divU;
				}
		}
	}
	
	template <typename Operator>
	void computeSplit(const double dt)
	{
		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();
		
#pragma omp parallel
		{
			Operator kernel(dt, minRho, *step);
			
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
	
	template <typename Operator>
	void compute(const double dt)
	{
		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();
		
#pragma omp parallel
		{
			Operator kernel(dt);
			
            Lab mylab;
#ifdef _MOVING_FRAME_
            mylab.pDirichlet.u = *uBody;
            mylab.pDirichlet.v = *vBody;
#endif
			mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, true);
			
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
		*pressureDragX = 0;
		*pressureDragY = 0;
		
		Real tmpDragX = 0;
		Real tmpDragY = 0;
		
#pragma omp parallel
		{
			OperatorPressureDrag kernel(0);
			
			Lab mylab;
#ifdef _MOVING_FRAME_
			mylab.pDirichlet.u = *uBody;
			mylab.pDirichlet.v = *vBody;
#endif
			mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, false);
			
#pragma omp for schedule(static) reduction(+:tmpDragX) reduction(+:tmpDragY)
			for (int i=0; i<N; i++)
			{
				mylab.load(ary[i], 0);
				kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
				tmpDragX += kernel.getDrag(0);
				tmpDragY += kernel.getDrag(1);
			}
		}
		
		*pressureDragX = tmpDragX;
		*pressureDragY = tmpDragY;
	}
	
public:
	CoordinatorPressure(const double minRho, const Real gravity[2], Real * uBody,
			Real * vBody, Real * pressureDragX, Real * pressureDragY, int * step,
			const bool bSplit, FluidGrid * grid, const int rank, const int nprocs)
: GenericCoordinator(grid), rank(rank), nprocs(nprocs), minRho(minRho), step(step),
  bSplit(bSplit), uBody(uBody), vBody(vBody), pressureDragX(pressureDragX),
  pressureDragY(pressureDragY), gravity{gravity[0],gravity[1]}, pressureSolver(NTHREADS,*grid)
	{
	}
    
    CoordinatorPressure(const double minRho, const Real gravity[2], int * step,
    		const bool bSplit, FluidGrid * grid, const int rank, const int nprocs)
    : GenericCoordinator(grid), rank(rank), nprocs(nprocs), minRho(minRho), step(step),
	  bSplit(bSplit), uBody(NULL), vBody(NULL), pressureDragX(NULL), pressureDragY(NULL),
	  gravity{gravity[0],gravity[1]}, pressureSolver(NTHREADS,*grid)
    {
    }
	
	void operator()(const double dt)
	{
		// need an interface that is the same for all solvers - this way the defines can be removed more cleanly
		
		check("pressure - start");
		
		// pressure
#ifdef _HYDROSTATIC_
		abort();
		addHydrostaticPressure(dt);
#endif // _HYDROSTATIC_
		computeSplit<OperatorDivergenceSplit>(dt);
		pressureSolver.solve(*grid,true);
		computeSplit<OperatorGradPSplit>(dt);
		updatePressure();
		
		drag();
		
		check("pressure - end");
	}
	
	string getName()
	{
		return "Pressure";
    }
};

template <typename Lab>
class CoordinatorPressureSimple : public GenericCoordinator
{
protected:
	Real *pressureDragX, *pressureDragY;
	
#ifndef _MIXED_
#ifndef _BOX_
#ifndef _OPENBOX_
    PoissonSolverScalarFFTW<FluidGrid, StreamerDiv> pressureSolver;
#else
	PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
#endif // _OPENBOX_
#else
	PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
#endif // _BOX_
#else
    PoissonSolverScalarFFTW_DCT<FluidGrid, StreamerDiv> pressureSolver;
#endif // _MIXED_
	
	inline void updatePressure()
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
					b(ix,iy).pOld = b(ix,iy).p;
					b(ix,iy).p    = b(ix,iy).divU;
				}
		}
	}
	
	template <typename Operator>
	void compute(const double dt)
	{
		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();
		
#pragma omp parallel
		{
			Operator kernel(dt);
			
			Lab mylab;
			mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, true);
			
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
		*pressureDragX = 0;
		*pressureDragY = 0;
		
		Real tmpDragX = 0;
		Real tmpDragY = 0;
		
#pragma omp parallel
		{
			OperatorPressureDrag kernel(0);
			
			Lab mylab;
#ifdef _MOVING_FRAME_
//			mylab.pDirichlet.u = 0;
//			mylab.pDirichlet.v = *vBody;
#endif
			mylab.prepare(*grid, kernel.stencil_start, kernel.stencil_end, false);
			
#pragma omp for schedule(static) reduction(+:tmpDragX) reduction(+:tmpDragY)
			for (int i=0; i<N; i++)
			{
				mylab.load(ary[i], 0);
				kernel(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
				tmpDragX += kernel.getDrag(0);
				tmpDragY += kernel.getDrag(1);
			}
		}
		
		*pressureDragX = tmpDragX;
		*pressureDragY = tmpDragY;
	}
	
public:
	CoordinatorPressureSimple(Real * pressureDragX, Real * pressureDragY, FluidGrid * grid) : GenericCoordinator(grid), pressureDragX(pressureDragX), pressureDragY(pressureDragY), pressureSolver(NTHREADS,*grid)
	{
	}
	
	CoordinatorPressureSimple(FluidGrid * grid) : GenericCoordinator(grid), pressureDragX(NULL), pressureDragY(NULL), pressureSolver(NTHREADS,*grid)
	{
	}
	
	void operator()(const double dt)
	{
		compute<OperatorDivergence>(dt);
		pressureSolver.solve(*grid,false);
		compute<OperatorGradP>(dt);
		
		updatePressure();
		
		drag();
	}
	
	string getName()
	{
		return "Pressure";
	}
};
#endif

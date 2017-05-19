//
//  TestConditionNumber.cpp
//  CubismUP_2D
//
//  Created by Christian Conti on 11/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "TestConditionNumber.h"

#include "ProcessOperatorsOMP.h"

#include "CoordinatorIC.h"


inline void TestConditionNumber::_mean(Real c, Real e, Real w, Real n, Real s, Real& avgE, Real& avgW, Real& avgN, Real& avgS)
{
	avgE = .5 * (c + e);
	avgW = .5 * (c + w);
	avgN = .5 * (c + n);
	avgS = .5 * (c + s);
}

void TestConditionNumber::_ic()
{
	path2file = parser("-file").asString("../data/ConditionNumber");
	
	Real centerOfMass[2] = {.5,.5};
    bool bPeriodic[2] = {false,false};
    
    vector<BlockInfo> vInfo = grid->getBlocksInfo();
    const Real domainSize[2] = { FluidBlock::sizeX * grid->getBlocksPerDimension(0) * vInfo[0].h_gridpoint,
								 FluidBlock::sizeY * grid->getBlocksPerDimension(1) * vInfo[0].h_gridpoint };
	
    Real radius = parser("-radius").asDouble(0.2);
	
	shape = new Disk(centerOfMass, radius, rhoS, 2, 2, bPeriodic, domainSize);
	
#ifdef _MULTIGRID_
	if (rank==0)
#endif // _MULTIGRID_
	{
		// setup initial conditions
		CoordinatorIC coordIC(shape,0,grid);
		coordIC(0);
	}
}

TestConditionNumber::TestConditionNumber(const int argc, const char ** argv, const int bpd) : Test(argc, argv), parser(argc,argv), bpd(bpd)
{
	// output settings
	path2file = parser("-file").asString("../data/TestConditionNumber");
	
	grid = new FluidGrid(bpd,bpd,1);
	
}

TestConditionNumber::~TestConditionNumber()
{
	delete grid;
}

void TestConditionNumber::run()
{
	const int stencil_start[3] = {-1,-1,0};
	const int stencil_end[3] = {2,2,1};
	
	// save variable coefficients Poisson in file
	vector<double> densities{0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 2, 10, 100, 1000, 10000};
	
	const int n = bpd*bpd*FluidBlock::sizeX*FluidBlock::sizeY;
	
	for (const double &d : densities)
	{
		double * matrix = new double[n*n];
		cout << "Density " << d << endl;
		rhoS = d;
		
		_ic();
		
		// fill
		vector<BlockInfo> vInfo = grid->getBlocksInfo();
		BlockInfo * ary = &vInfo.front();
		const int N = vInfo.size();
		
#pragma omp parallel
		{
			Lab lab;
			lab.prepare(*grid, stencil_start, stencil_end, false);
			
#pragma omp for schedule(static)
			for (int i=0; i<N; i++)
			{
				BlockInfo info = vInfo[i];
				lab.load(ary[i], 0);
				
				for(int iy=0; iy<FluidBlock::sizeY; ++iy)
					for(int ix=0; ix<FluidBlock::sizeX; ++ix)
					{
						const int size = bpd*FluidBlock::sizeX;
						const int gix = ix + info.index[0]*FluidBlock::sizeX;
						const int giy = iy + info.index[1]*FluidBlock::sizeY;
						const int idx = gix + giy*size;
						
						if (gix >= n || giy >= n)
						{
							cout << "index out of range\n";
							abort();
						}
						
						Real rhoS, rhoN, rhoE, rhoW;
						
						FluidElement& phi  = lab(ix  ,iy  );
						FluidElement& phiN = lab(ix  ,iy+1);
						FluidElement& phiS = lab(ix  ,iy-1);
						FluidElement& phiE = lab(ix+1,iy  );
						FluidElement& phiW = lab(ix-1,iy  );
						
						_mean(phi.rho, phiE.rho, phiW.rho, phiN.rho, phiS.rho, rhoE, rhoW, rhoN, rhoS);
						
						if (gix>0 && gix<bpd*FluidBlock::sizeX-1 && giy>0 && giy<bpd*FluidBlock::sizeY-1)
						{
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN + 1./rhoS);
							matrix[idx+1    + n*idx] = 1./rhoE;
							matrix[idx-1    + n*idx] = 1./rhoW;
							matrix[idx+size + n*idx] = 1./rhoN;
							matrix[idx-size + n*idx] = 1./rhoS;
						}
						else if (gix==0 && giy>0 && giy<bpd*FluidBlock::sizeY-1)
						{
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN + 1./rhoS);
							matrix[idx+1      + n*idx] = 1./rhoE;
							matrix[idx+size-1 + n*idx] = 1./rhoW; // periodic
							matrix[idx+size   + n*idx] = 1./rhoN;
							matrix[idx-size   + n*idx] = 1./rhoS;
						}
						else if (gix==bpd*FluidBlock::sizeX-1 && giy>0 && giy<bpd*FluidBlock::sizeY-1)
						{
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN + 1./rhoS);
							matrix[idx-size+1 + n*idx] = 1./rhoE; // periodic
							matrix[idx-1      + n*idx] = 1./rhoW;
							matrix[idx+size   + n*idx] = 1./rhoN;
							matrix[idx-size   + n*idx] = 1./rhoS;
						}
						else if (giy==0 && gix>0 && gix<bpd*FluidBlock::sizeX-1)
						{
							// Neumann
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN);
							matrix[idx+1    + n*idx] = 1./rhoE;
							matrix[idx-1    + n*idx] = 1./rhoW;
							matrix[idx+size + n*idx] = 1./rhoN;
						}
						else if (giy==bpd*FluidBlock::sizeY-1 && gix>0 && gix<bpd*FluidBlock::sizeX-1)
						{
							// Dirichlet
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN + 1./rhoS);
							matrix[idx+1    + n*idx] = 1./rhoE;
							matrix[idx-1    + n*idx] = 1./rhoW;
							matrix[idx-size + n*idx] = 1./rhoS;
						}
						else if (gix==0 && giy==0)
						{
							// corners
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN);
							matrix[idx+1      + n*idx] = 1./rhoE;
							matrix[idx+size-1 + n*idx] = 1./rhoW; // periodic
							matrix[idx+size   + n*idx] = 1./rhoN;
						}
						else if (gix==0 && giy==bpd*FluidBlock::sizeY-1)
						{
							// corners
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN + 1./rhoS);
							matrix[idx+1      + n*idx] = 1./rhoE;
							matrix[idx+size-1 + n*idx] = 1./rhoW; // periodic
							matrix[idx-size   + n*idx] = 1./rhoS;
						}
						else if (gix==bpd*FluidBlock::sizeX-1 && giy==0)
						{
							// corners
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN);
							matrix[idx-size+1 + n*idx] = 1./rhoE;
							matrix[idx-1      + n*idx] = 1./rhoW; // periodic
							matrix[idx+size   + n*idx] = 1./rhoN;
						}
						else if (gix==bpd*FluidBlock::sizeX-1 && giy==bpd*FluidBlock::sizeY-1)
						{
							// corners
							matrix[idx + n*idx] = -(1./rhoE + 1./rhoW + 1./rhoN + 1./rhoS);
							matrix[idx-size+1 + n*idx] = 1./rhoE;
							matrix[idx-1      + n*idx] = 1./rhoW; // periodic
							matrix[idx-size   + n*idx] = 1./rhoS;
						}
					}
				
			}
		}
		
		// dump
		stringstream ss;
		ss << path2file << d << "_bpd" << bpd << ".dat";
		ofstream myfile(ss.str(), fstream::app);
		for (int j=0; j<n; j++)
		{
			for (int i=0; i<n; i++)
			{
				myfile << matrix[i+j*n] << " ";
			}
			myfile << endl;
		}
		 
		delete shape;
		delete [] matrix;
	}
}

void TestConditionNumber::check()
{
}
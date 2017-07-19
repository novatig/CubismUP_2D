//
//  Simulation_Fluid.h
//  CubismUP_2D
//
//	Base class for fluid simulations from which any fluid simulation case should inherit
//	Contains the base structure and interface that any fluid simulation class should have
//
//  Created by Christian Conti on 3/25/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_Simulation_Fluid_h
#define CubismUP_2D_Simulation_Fluid_h

#include "Definitions.h"
//#include "ProcessOperatorsOMP.h"
#include "GenericCoordinator.h"
#include "GenericOperator.h"

#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

class Simulation_Fluid
{
 protected:
	ArgumentParser parser;
	Profiler profiler;

	// Serialization
	bool bPing; // needed for ping-pong scheme
	string path4serialization;
	bool bRestart;

	// MPI stuff - required for Hypre
	int rank, nprocs;

	vector<GenericCoordinator *> pipeline;

	// grid
	int bpdx, bpdy;
	FluidGrid * grid;

	// simulation status
	int step, nsteps;
	double dt, time, endTime;

	// simulation settings
	double CFL, LCFL;

	// verbose
	bool verbose;

	// output
	int dumpFreq;
	double dumpTime;
	string path2file;
	SerializerIO_ImageVTK<FluidGrid, FluidVTKStreamer> dumper;

	virtual void _diagnostics() = 0;
	virtual void _ic() = 0;
	virtual double _nonDimensionalTime() = 0;

	virtual void _dump(double & nextDumpTime)
	{
		#ifndef NDEBUG
		{
			vector<BlockInfo> vInfo = grid->getBlocksInfo();
			const int N = vInfo.size();

			#pragma omp parallel for schedule(static)
			for(int i=0; i<N; i++)
			{
				BlockInfo info = vInfo[i];
				FluidBlock& b = *(FluidBlock*)info.ptrBlock;

				for(int iy=0; iy<FluidBlock::sizeY; ++iy)
					for(int ix=0; ix<FluidBlock::sizeX; ++ix)
					{
						if (std::isnan(b(ix,iy).rho) ||
							std::isnan(b(ix,iy).u) ||
							std::isnan(b(ix,iy).v) ||
							std::isnan(b(ix,iy).chi) ||
							std::isnan(b(ix,iy).p) ||
							std::isnan(b(ix,iy).pOld))
							cout << "dump" << endl;

						if (b(ix,iy).rho <= 0)
							cout << "dump " << b(ix,iy).rho << "\t" << info.index[0] << " " << info.index[1] << " " << ix << " " << iy << endl;

						assert(b(ix,iy).rho > 0);
						assert(!std::isnan(b(ix,iy).rho));
						assert(!std::isnan(b(ix,iy).u));
						assert(!std::isnan(b(ix,iy).v));
						assert(!std::isnan(b(ix,iy).chi));
						assert(!std::isnan(b(ix,iy).p));
						assert(!std::isnan(b(ix,iy).pOld));
						assert(!std::isnan(b(ix,iy).tmpU));
						assert(!std::isnan(b(ix,iy).tmpV));
						assert(!std::isnan(b(ix,iy).tmp));
					}
			}
		}
		#endif

		const int sizeX = bpdx * FluidBlock::sizeX;
		const int sizeY = bpdy * FluidBlock::sizeY;
		vector<BlockInfo> vInfo = grid->getBlocksInfo();

		const bool timeDump = dumpTime>0. && _nonDimensionalTime()>nextDumpTime;
		const bool stepDump = dumpFreq>0  && step % dumpFreq == 0;
		if(stepDump || timeDump)
		{
			nextDumpTime += dumpTime;

			//vector<BlockInfo> vInfo = grid->getBlocksInfo();
			//processOMP<Lab, OperatorVorticityTmp>(0, vInfo,*grid);
			stringstream ss;
      ss << path2file << "avemaria_";
      ss << std::setfill('0') << std::setw(7) << step;
      ss << ".vti";
			cout << ss.str() << endl;

			dumper.Write(*grid, ss.str());
			//_serialize();
		}
	}

	virtual void _dump(stringstream&fname)
	{
		const int sizeX = bpdx * FluidBlock::sizeX;
		const int sizeY = bpdy * FluidBlock::sizeY;
		//vector<BlockInfo> vInfo = grid->getBlocksInfo();
		//processOMP<Lab, OperatorVorticityTmp>(0, vInfo,*grid);
    cout << fname.str() << endl;
		dumper.Write(*grid, fname.str());
		_serialize();
	}

	void _serialize()
	{
		stringstream ss;
		ss << path4serialization << "Serialized-" << bPing << ".dat";
		cout << ss.str() << endl;

		stringstream serializedGrid;
		serializedGrid << "SerializedGrid-" << bPing << ".grid";
		DumpZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);

		bPing = !bPing;
	}

	void _deserialize()
	{
		/*
		stringstream ss0, ss1, ss;
		struct stat st0, st1;
		ss0 << path4serialization << "Serialized-0.dat";
		ss1 << path4serialization << "Serialized-1.dat";
		stat(ss0.str().c_str(), &st0);
		stat(ss1.str().c_str(), &st1);

		grid = new FluidGrid(bpdx,bpdy,1);
		assert(grid != NULL);
		{
			stringstream serializedGrid;
			serializedGrid << "SerializedGrid-" << bPing << ".grid";
			ReadZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
		}
		*/
	}

 public:
	Simulation_Fluid(const int argc, const char ** argv) :
		parser(argc,argv), step(0), time(0), dt(0), rank(0), nprocs(1), bPing(false)
	{
	}

	virtual ~Simulation_Fluid()
	{
		delete grid;

		while(!pipeline.empty())
		{
			GenericCoordinator * g = pipeline.back();
			pipeline.pop_back();
			delete g;
		}
	}

	virtual void init()
	{
		bRestart = parser("-restart").asBool(false);
		cout << "bRestart is " << bRestart << endl;

		// initialize grid
		parser.set_strict_mode();
		bpdx = parser("-bpdx").asInt();
		bpdy = parser("-bpdy").asInt();
		grid = new FluidGrid(bpdx,bpdy,1);
		assert(grid != NULL);

		// simulation ending parameters
		parser.unset_strict_mode();
		nsteps = parser("-nsteps").asInt(0);		// nsteps==0   means that this stopping criteria is not active
		endTime = parser("-tend").asDouble(0);		// endTime==0  means that this stopping criteria is not active

		// output parameters
		dumpFreq = parser("-fdump").asDouble(0);	// dumpFreq==0 means that this dumping frequency (in #steps) is not active
		dumpTime = parser("-tdump").asDouble(0);	// dumpTime==0 means that this dumping frequency (in time)   is not active
		path2file = parser("-file").asString("./");
		path4serialization = parser("-serialization").asString(path2file);

		CFL = parser("-CFL").asDouble(.1);
		LCFL = parser("-LCFL").asDouble(.1);

		verbose = parser("-verbose").asBool(false);
	}

	virtual void simulate() = 0;
};

#endif

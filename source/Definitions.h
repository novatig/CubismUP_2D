//
//  DataStructures.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/7/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_DataStructures_h
#define CubismUP_2D_DataStructures_h

#include "common.h"
#include "Layer.h"
#include "LayerToVTK.h"
#include "BoundaryConditions.h"

#ifndef _BS_
#define _BS_ 32
#endif // _BS_

struct ObstacleBlock
{
  static const int sizeX = _BS_;
  static const int sizeY = _BS_;
  Real chi[sizeX][sizeY];
  Real rho[sizeX][sizeY];
  Real udef[sizeX][sizeY][2];

  void clear()
  {
    memset(chi,  0, sizeof(Real)*sizeX*sizeY);
    memset(udef, 0, sizeof(Real)*sizeX*sizeY*2);
		for(int i=0; i<sizeX*sizeY; i++) rho[i] = 1;
  }
};

struct FluidElement
{
  Real rho, u, v, p, pOld;
	Real tmpU, tmpV, tmp;

    FluidElement() :
    rho(0), u(0), v(0), p(0), pOld(0), tmpU(0), tmpV(0), tmp(0)//, divU(0), x(0), y(0)
    {}

    void clear()
    {
        rho = u = v = p = pOld = tmpU = tmpV = tmp = 0;
    }
};


struct FluidVTKStreamer
{
	//static const int channels = 6;
	static const int channels = 4;

	void operate(FluidElement input, Real output[channels])
	{
		output[0] = input.rho;
		output[1] = input.u;
		output[2] = input.v;
		output[3] = input.p;
		//output[k++] = input.chi;
		//output[k++] = input.tmp;
	}
};

// this is used for serialization - important that ALL the quantities are streamed
struct StreamerGridPoint
{
	static const int channels = 9;

	void operate(const FluidElement& input, Real output[channels]) const
	{
		abort();
		output[0] = input.rho;
		output[1] = input.u;
		output[2] = input.v;
		output[3] = input.p;
		output[4] = input.pOld;
		output[5] = input.tmpU;
		output[6] = input.tmpV;
		output[7] = input.tmp;
	}

	void operate(const Real input[channels], FluidElement& output) const
	{
		abort();
		output.rho  = input[0];
		output.u    = input[1];
		output.v    = input[2];
		output.p    = input[3];
		output.pOld = input[4];
		output.tmpU = input[5];
		output.tmpV = input[6];
		output.tmp  = input[7];
	}
};

struct StreamerGridPointASCII
{
	void operate(const FluidElement& input, ofstream& output) const
	{
		output << input.rho << " " << input.u << " " << input.v << " " << input.p << " " << input.pOld << " " << input.tmpU << " " << input.tmpV << " " << input.tmp;
	}

	void operate(ifstream& input, FluidElement& output) const
	{
		input >> output.rho;
		input >> output.u;
		input >> output.v;
		input >> output.p;
		input >> output.pOld;
		input >> output.tmpU;
		input >> output.tmpV;
		input >> output.tmp;
	}
};

struct StreamerDiv
{
	static const int channels = 1;
	static void operate(const FluidElement& input, Real output[1])
	{
	   output[0] = input.tmp;
  }

  static void operate(const Real input[1], FluidElement& output)
  {
      output.tmp = input[0];
  }
};

struct FluidBlock
{
    //these identifiers are required by cubism!
    static const int sizeX = _BS_;
    static const int sizeY = _BS_;
    static const int sizeZ = 1;
    typedef FluidElement ElementType;
    FluidElement data[1][sizeY][sizeX];

    //required from Grid.h
    void clear()
    {
        FluidElement * entry = &data[0][0][0];
        const int N = sizeX*sizeY;

        for(int i=0; i<N; ++i)
            entry[i].clear();
    }

    FluidElement& operator()(int ix, int iy=0, int iz=0)
    {
        assert(ix>=0); assert(ix<sizeX);
        assert(iy>=0); assert(iy<sizeY);

        return data[0][iy][ix];
	}

	template <typename Streamer>
	inline void Write(ofstream& output, Streamer streamer) const
	{
		for(int iy=0; iy<sizeY; iy++)
			for(int ix=0; ix<sizeX; ix++)
				streamer.operate(data[0][iy][ix], output);
	}

	template <typename Streamer>
	inline void Read(ifstream& input, Streamer streamer)
	{
		for(int iy=0; iy<sizeY; iy++)
			for(int ix=0; ix<sizeX; ix++)
				streamer.operate(input, data[0][iy][ix]);
	}
};

template <> inline void FluidBlock::Write<StreamerGridPoint>(ofstream& output, StreamerGridPoint streamer) const
{
	output.write((const char *)&data[0][0][0], sizeof(FluidElement)*sizeX*sizeY);
}

template <> inline void FluidBlock::Read<StreamerGridPoint>(ifstream& input, StreamerGridPoint streamer)
{
	input.read((char *)&data[0][0][0], sizeof(FluidElement)*sizeX*sizeY);
}



struct StreamerSerialization
{
	static const int NCHANNELS = 5;

	FluidBlock& ref;

	StreamerSerialization(FluidBlock& b): ref(b) {}

	void operate(const int ix, const int iy, const int iz, Real output[NCHANNELS]) const
	{
		const FluidElement& input = ref.data[iz][iy][ix];
		output[0] = input.rho;
		output[1] = input.u;
		output[2] = input.v;
		output[3] = input.p;
		output[4] = input.pOld;
		//output[6] = input.tmpU;
		//output[7] = input.tmpV;
		//output[8] = input.tmp;
		//output[9] = input.divU;
	}

	void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz) const
	{
		FluidElement& output = ref.data[iz][iy][ix];
		output.rho  = input[0];
		output.u    = input[1];
		output.v    = input[2];
		output.p    = input[3];
		output.pOld = input[4];
		//output.tmpU = input[6];
		//output.tmpV = input[7];
		//output.tmp  = input[8];
		//output.divU = input[9];
	}

	void operate(const int ix, const int iy, const int iz, Real *ovalue, const int field) const
	{
		const FluidElement& input = ref.data[iz][iy][ix];

		switch(field) {
			case 0: *ovalue = input.rho; break;
			case 1: *ovalue = input.u; break;
			case 2: *ovalue = input.v; break;
			case 3: *ovalue = input.p; break;
			case 4: *ovalue = input.pOld; break;
			//case 6: *ovalue = input.tmpU; break;
			//case 7: *ovalue = input.tmpV; break;
			//case 8: *ovalue = input.tmp; break;
			//case 9: *ovalue = input.divU; break;
			default: throw std::invalid_argument("unknown field!"); break;
		}
	}

	void operate(const Real ivalue, const int ix, const int iy, const int iz, const int field) const
	{
		FluidElement& output = ref.data[iz][iy][ix];

		switch(field) {
			case 0:  output.rho  = ivalue; break;
			case 1:  output.u    = ivalue; break;
			case 2:  output.v    = ivalue; break;
			case 3:  output.p    = ivalue; break;
			case 4:  output.pOld = ivalue; break;
			//case 6:  output.tmpU = ivalue; break;
			//case 7:  output.tmpV = ivalue; break;
			//case 8:  output.tmp  = ivalue; break;
			//case 9:  output.divU = ivalue; break;
			default: throw std::invalid_argument("unknown field!"); break;
		}
	}

	static const char * getAttributeName() { return "Tensor"; }
};

template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabBottomWall : public BlockLab<BlockType,allocator>
{
	typedef typename BlockType::ElementType ElementTypeBlock;

 public:
    ElementTypeBlock pDirichlet;

	BlockLabBottomWall(): BlockLab<BlockType,allocator>()
    {
        pDirichlet.rho = 1;
        pDirichlet.u = 0;
        pDirichlet.v = 0;
        pDirichlet.p = 0;
        pDirichlet.pOld = 0;
        //pDirichlet.divU = 0;
        pDirichlet.tmp = 1;
        pDirichlet.tmpU = 0;
        pDirichlet.tmpV = 0;
    }

	void _apply_bc(const BlockInfo& info, const Real t=0)
	{
		BoundaryCondition<BlockType,ElementTypeBlock,allocator> bc(this->m_stencilStart, this->m_stencilEnd, this->m_cacheBlock);

		// keep periodicity in x direction
		if (info.index[1]==0)		   bc.template applyBC_mixedBottom<1,0>(pDirichlet);
		if (info.index[1]==this->NY-1) bc.template applyBC_mixedTop<1,1>(pDirichlet);
	}
};

template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabPipe : public BlockLab<BlockType,allocator>
{
	typedef typename BlockType::ElementType ElementTypeBlock;

 public:
	BlockLabPipe(): BlockLab<BlockType,allocator>(){}

	void _apply_bc(const BlockInfo& info, const Real t=0)
	{
		BoundaryCondition<BlockType,ElementTypeBlock,allocator> bc(this->m_stencilStart, this->m_stencilEnd, this->m_cacheBlock);

		if (info.index[1]==0)		   bc.template applyBC_mixedBottom<1,0>();
		if (info.index[1]==this->NY-1) bc.template applyBC_mixedBottom<1,1>();
	}
};

template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabVortex : public BlockLab<BlockType,allocator>
{
	typedef typename BlockType::ElementType ElementTypeBlock;

 public:
	BlockLabVortex(): BlockLab<BlockType,allocator>(){}

	void _apply_bc(const BlockInfo& info, const Real t=0)
	{
		BoundaryCondition<BlockType,ElementTypeBlock,allocator> bc(this->m_stencilStart, this->m_stencilEnd, this->m_cacheBlock);

		if (info.index[0]==0)		   bc.template applyBC_vortex<0,0>(info);
		if (info.index[0]==this->NX-1) bc.template applyBC_vortex<0,1>(info);
		if (info.index[1]==0)		   bc.template applyBC_vortex<1,0>(info);
		if (info.index[1]==this->NY-1) bc.template applyBC_vortex<1,1>(info);
	}
};

template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabOpenBox : public BlockLab<BlockType,allocator>
{
	typedef typename BlockType::ElementType ElementTypeBlock;

 public:
	ElementTypeBlock pDirichlet;

	BlockLabOpenBox(): BlockLab<BlockType,allocator>()
	{
		pDirichlet.rho = 1;
		pDirichlet.u = 0;
		pDirichlet.v = 0;
		pDirichlet.p = 0;
		pDirichlet.pOld = 0;
		//pDirichlet.divU = 0;
		pDirichlet.tmp = 1;
		pDirichlet.tmpU = 0;
		pDirichlet.tmpV = 0;
	}

	void _apply_bc(const BlockInfo& info, const Real t=0)
	{
		BoundaryCondition<BlockType,ElementTypeBlock,allocator> bc(this->m_stencilStart, this->m_stencilEnd, this->m_cacheBlock);

		if (info.index[0]==0)		   bc.template applyBC_BoxLeft<0,0>(pDirichlet);
		if (info.index[0]==this->NX-1) bc.template applyBC_BoxRight<0,1>(pDirichlet);
		if (info.index[1]==0)		   bc.template applyBC_mixedBottom<1,0>(pDirichlet);
		if (info.index[1]==this->NY-1) bc.template applyBC_mixedTop<1,1>(pDirichlet);
	}
};

template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabBox : public BlockLab<BlockType,allocator>
{
	typedef typename BlockType::ElementType ElementTypeBlock;

 public:
	ElementTypeBlock pDirichlet;

	BlockLabBox(): BlockLab<BlockType,allocator>()
	{
		pDirichlet.rho = 1;
		pDirichlet.u = 0;
		pDirichlet.v = 0;
		pDirichlet.p = 0;
		pDirichlet.pOld = 0;
		//pDirichlet.divU = 0;
		pDirichlet.tmp = 1;
		pDirichlet.tmpU = 0;
		pDirichlet.tmpV = 0;
	}

	void _apply_bc(const BlockInfo& info, const Real t=0)
	{
		BoundaryCondition<BlockType,ElementTypeBlock,allocator> bc(this->m_stencilStart, this->m_stencilEnd, this->m_cacheBlock);

		if (info.index[0]==0)		   bc.template applyBC_BoxLeft<0,0>(pDirichlet);
		if (info.index[0]==this->NX-1) bc.template applyBC_BoxRight<0,1>(pDirichlet);
		if (info.index[1]==0)		   bc.template applyBC_mixedBottom<1,0>(pDirichlet);
		if (info.index[1]==this->NY-1) bc.template applyBC_BoxTop<1,1>(pDirichlet);
	}
};

template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabOpen: public BlockLab<BlockType,allocator>
{
    typedef typename BlockType::ElementType ElementTypeBlock;

  public:
    ElementTypeBlock pDirichlet;
    BlockLabOpen(): BlockLab<BlockType,allocator>(){}
    void _apply_bc(const BlockInfo& info, const Real t=0)
    {
        BoundaryCondition<BlockType,ElementTypeBlock,allocator>
                bc(this->m_stencilStart, this->m_stencilEnd, this->m_cacheBlock);

        if (info.index[0]==0)           bc.template applyBC_absorbing<0,0>();
        if (info.index[0]==this->NX-1)  bc.template applyBC_absorbing<0,1>();
        if (info.index[1]==0)           bc.template applyBC_absorbing<1,0>();
        if (info.index[1]==this->NY-1)  bc.template applyBC_absorbing<1,1>();
    }
};

typedef Grid<FluidBlock, std::allocator> FluidGrid;

#ifdef _MIXED_
typedef BlockLabBottomWall<FluidBlock, std::allocator> Lab;
#endif // _MIXED_

#ifdef _PERIODIC_
//typedef BlockLab<FluidBlock, std::allocator> Lab;
typedef BlockLabOpen<FluidBlock, std::allocator> Lab;
#endif // _PERIODIC_

#ifdef _VORTEX_
typedef BlockLabVortex<FluidBlock, std::allocator> Lab;
#endif // _VORTEX_

#ifdef _PIPE_
typedef BlockLabPipe<FluidBlock, std::allocator> Lab;
#endif // _PIPE_

#ifdef _OPENBOX_
typedef BlockLabOpenBox<FluidBlock, std::allocator> Lab;
#endif // _PIPE_

#ifdef _BOX_
typedef BlockLabBox<FluidBlock, std::allocator> Lab;
#endif // _PIPE_

#endif

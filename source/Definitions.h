//
//  DataStructures.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/7/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "common.h"
#include "BoundaryConditions.h"

#ifndef _BS_
#define _BS_ 32
#endif // _BS_

struct FluidElement
{
  Real u, v, tmpU, tmpV; //used by advection and diffusion
  Real invRho, tmp, p, pOld; //used by pressure

  FluidElement() :
  u(0), v(0), tmpU(0), tmpV(0), invRho(0), tmp(0), p(0), pOld(0)
  {}

  void clear()
  {
    u = v = tmpU = tmpV = invRho = tmp = p = pOld = 0;
  }
};

struct FluidVTKStreamer
{
  static const int channels = 5;

  void operate(FluidElement input, Real output[channels])
  {
    output[0] = 1/input.invRho;
    output[1] = input.u;
    output[2] = input.v;
    output[3] = input.p;
    output[4] = input.tmp;
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

// this is used for serialization - important that ALL the quantities are treamed
struct StreamerGridPoint
{
  static const int channels = 9;
  void operate(const FluidElement& input, Real output[channels]) const
  {
    abort();
    output[0] = input.invRho;
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
    output.invRho  = input[0];
    output.u    = input[1];
    output.v    = input[2];
    output.p    = input[3];
    output.pOld = input[4];
    output.tmpU = input[5];
    output.tmpV = input[6];
    output.tmp  = input[7];
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
    output[0] = input.invRho;
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
    output.invRho  = input[0];
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
      case 0: *ovalue = input.invRho; break;
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
      case 0:  output.invRho  = ivalue; break;
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


struct StreamerChi
{
    static const int NCHANNELS = 1;
    static const int CLASS = 0;

    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
      output[0] = b(ix,iy,iz).chi;
    }
    static std::string prefix()
    {
      return std::string("chi_");
    }

    static const char * getAttributeName() { return "Scalar"; }
};

struct StreamerVelocityVector
{
    static const int NCHANNELS = 3;
    static const int CLASS = 0;
    // Write
    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
        output[0] = b(ix,iy,iz).u;
        output[1] = b(ix,iy,iz).v;
        output[2] = b(ix,iy,iz).tmp;
    }
    // Read
    template <typename TBlock, typename T>
    static inline void operate(TBlock& b, const T input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        b(ix,iy,iz).u = input[0];
        b(ix,iy,iz).v = input[1];
    }
    static std::string prefix()
    {
      return std::string("vel_");
    }
    static const char * getAttributeName() { return "Vector"; }
};

struct StreamerTmpVector
{
    static const int NCHANNELS = 3;
    static const int CLASS = 0;
    // Write
    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
        output[0] = b(ix,iy,iz).tmpU;
        output[1] = b(ix,iy,iz).tmpV;
        output[2] = b(ix,iy,iz).tmp;
    }
    static std::string prefix()
    {
      return std::string("tmp_");
    }
    static const char * getAttributeName() { return "Vector"; }
};

struct StreamerPressure
{
    static const int NCHANNELS = 1;
    static const int CLASS = 0;
    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
      output[0] = b(ix,iy,iz).p;
    }
    static std::string prefix()
    {
      return std::string("pres_");
    }
    static const char * getAttributeName() { return "Scalar"; }
};
struct StreamerRho
{
    static const int NCHANNELS = 1;
    static const int CLASS = 0;
    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
      output[0] = b(ix,iy,iz).invRho;
    }
    static std::string prefix()
    {
      return std::string("invRho_");
    }
    static const char * getAttributeName() { return "Scalar"; }
};

typedef Grid<FluidBlock, std::allocator> FluidGrid;

//#ifndef _MIXED_
//#ifndef _BOX_
//#ifndef _OPENBOX_
//typedef BlockLab<FluidBlock, std::allocator> Lab;
typedef BlockLabOpen<FluidBlock, std::allocator> Lab;
//#else
//  typedef BlockLabOpenBox<FluidBlock, std::allocator> Lab;
//#endif // _OPENBOX_
//#else
//  typedef BlockLabBox<FluidBlock, std::allocator> Lab;
//#endif // _BOX_
//#else
//  typedef BlockLabBottomWall<FluidBlock, std::allocator> Lab;
//#endif // _MIXED_

class Shape;

struct SimulationData
{
  FluidGrid * grid = nullptr;
  vector<Shape*> shapes;

  double time = 0;
  int step = 0;

  Real uinfx = 0;
  Real uinfy = 0;

  double lambda = 0;
  double nu = 0;
  double dlm = -1;

  Real gravity[2] = { (Real) 0.0, (Real) -9.81 };
  // nsteps==0 means that this stopping criteria is not active
  int nsteps = 0;
  // endTime==0  means that this stopping criteria is not active
  double endTime = 0;

  double dt = 0;
  double CFL = 0.1;

  bool verbose = true;
  bool muteAll = false;
  bool bFreeSpace = true;
  // output
  // dumpFreq==0 means that this dumping frequency (in #steps) is not active
  int dumpFreq = 0;
  // dumpTime==0 means that this dumping frequency (in time)   is not active
  double dumpTime = 0;
  double nextDumpTime = 0;
  bool _bDump = false;
  bool bPing = false;
  bool bRestart = false;

  string path4serialization;
  string path2file;

  void resetAll();
  bool bDump()
  {
    const bool timeDump = dumpTime>0 && time > nextDumpTime;
    const bool stepDump = dumpFreq>0 && step % dumpFreq == 0;
    _bDump = stepDump || timeDump;
    return _bDump;
  }
  void registerDump();
  bool bOver() const
  {
    const bool timeEnd = endTime>0 && time > endTime;
    const bool stepEnd =  nsteps>0 && step > nsteps;
    return timeEnd || stepEnd;
  }
  double minRho() const;
  double maxSpeed() const;
  double maxRelSpeed() const;
  double getH() const
  {
    return grid->getBlocksInfo().front().h_gridpoint; // yikes
  }

  ~SimulationData();
};

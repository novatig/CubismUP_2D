//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>

//using namespace std;

#ifndef _FLOAT_PRECISION_
using Real = double;
#else // _FLOAT_PRECISION_
using Real = float;
#endif // _FLOAT_PRECISION_

#include <ArgumentParser.h>
#include <Grid.h>
#include <BlockInfo.h>
#include <BlockLab.h>
#include <StencilInfo.h>

#ifndef _BS_
#define _BS_ 32
#endif//_BS_

#ifndef _DIM_
#define _DIM_ 2
#endif//_BS_

struct ScalarElement {
  Real s = 0;
  inline void clear() { s = 0; }
  inline void set(const Real v) { s = v; }
  inline void copy(const ScalarElement& c) { s = c.s; }
  ScalarElement(const ScalarElement& c) = delete;
  ScalarElement& operator=(const ScalarElement& c) { s = c.s; return *this; }
};

struct VectorElement {
  static constexpr int DIM = _DIM_;
  Real u[DIM];

  VectorElement() { clear(); }

  inline void clear() { for(int i=0; i<DIM; ++i) u[i] = 0; }
  inline void set(const Real v) { for(int i=0; i<DIM; ++i) u[i] = v; }
  inline void copy(const VectorElement& c) {
    for(int i=0; i<DIM; ++i) u[i] = c.u[i];
  }
  VectorElement(const VectorElement& c) = delete;
  VectorElement& operator=(const VectorElement& c) {
    for(int i=0; i<DIM; ++i) u[i] = c.u[i];
    return *this;
  }
};

template <typename Element>
struct GridBlock
{
  static constexpr int sizeX = _BS_;
  static constexpr int sizeY = _BS_;
  static constexpr int sizeZ = _DIM_ > 2 ? _BS_ : 1;

  using ElementType = Element;
  alignas(32) ElementType data[sizeZ][sizeY][sizeX];

  inline void clear() {
      ElementType * const entry = &data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].clear();
  }
  inline void set(const Real v) {
      ElementType * const entry = &data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].set(v);
  }
  inline void copy(const GridBlock<Element>& c) {
      ElementType * const entry = &data[0][0][0];
      const ElementType * const source = &c.data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].copy(source[i]);
  }

  const ElementType& operator()(int ix, int iy=0, int iz=0) const {
      assert(ix>=0 && iy>=0 && iz>=0 && ix<sizeX && iy<sizeY && iz<sizeZ);
      return data[iz][iy][ix];
  }
  ElementType& operator()(int ix, int iy=0, int iz=0) {
      assert(ix>=0 && iy>=0 && iz>=0 && ix<sizeX && iy<sizeY && iz<sizeZ);
      return data[iz][iy][ix];
  }
  GridBlock(const GridBlock&) = delete;
  GridBlock& operator=(const GridBlock&) = delete;
};

template<typename BlockType,
         template<typename X> class allocator = std::allocator>
class BlockLabOpen: public BlockLab<BlockType, allocator>
{
 public:
  using ElementType = typename BlockType::ElementType;
  static constexpr int sizeX = BlockType::sizeX;
  static constexpr int sizeY = BlockType::sizeY;
  static constexpr int sizeZ = BlockType::sizeZ;

  // Used for Boundary Conditions:

  // Apply bc on face of direction dir and side side (0 or 1):
  template<int dir, int side> void applyBCface()
  {
    auto * const cb = this->m_cacheBlock;

    int s[3] = {0,0,0}, e[3] = {0,0,0};
    const int* const stenBeg = this->m_stencilStart;
    const int* const stenEnd = this->m_stencilEnd;
    s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX ) : stenBeg[0];
    s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY ) : stenBeg[1];

    e[0] =  dir==0 ? (side==0 ? 0 : sizeX + stenEnd[0]-1 )
                   : sizeX +  stenEnd[0]-1;
    e[1] =  dir==1 ? (side==0 ? 0 : sizeY + stenEnd[1]-1 )
                   : sizeY +  stenEnd[1]-1;

    for(int iy=s[1]; iy<e[1]; iy++)
    for(int ix=s[0]; ix<e[0]; ix++)
      cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0) = cb->Access
        (
          ( dir==0? (side==0? 0: sizeX-1):ix ) - stenBeg[0],
          ( dir==1? (side==0? 0: sizeY-1):iy ) - stenBeg[1],
          0
        );
  }

  // Called by Cubism:
  void _apply_bc(const BlockInfo& info, const Real t = 0)
  {
    if( info.index[0]==0 )           this->template applyBCface<0,0>();
    if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>();
    if( info.index[1]==0 )           this->template applyBCface<1,0>();
    if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>();
  }

  BlockLabOpen(): BlockLab<BlockType,allocator>(){}
  BlockLabOpen(const BlockLabOpen&) = delete;
  BlockLabOpen& operator=(const BlockLabOpen&) = delete;
};

struct StreamerScalar {
  static constexpr int NCHANNELS = 1;
  template <typename TBlock, typename T>
  static inline void operate(const TBlock& b,
    const int ix, const int iy, const int iz, T output[NCHANNELS]) {
    output[0] = b(ix,iy,iz).s;
  }
  static std::string prefix() { return std::string(""); }
  static const char * getAttributeName() { return "Scalar"; }
};

struct StreamerVector {
  static constexpr int NCHANNELS = 3;

  template <typename TBlock, typename T>
  static inline void operate(const TBlock& b,
    const int ix, const int iy, const int iz, T output[NCHANNELS]) {
      for (int i = 0; i < _DIM_; i++) output[i] = b(ix,iy,iz).u[i];
  }

  template <typename TBlock, typename T>
  static inline void operate(TBlock& b, const T input[NCHANNELS],
    const int ix, const int iy, const int iz) {
      for (int i = 0; i < _DIM_; i++) b(ix,iy,iz).u[i] = input[i];
  }
  static std::string prefix() { return std::string(""); }
  static const char * getAttributeName() { return "Vector"; }
};

using ScalarBlock = GridBlock<ScalarElement>;
using VectorBlock = GridBlock<VectorElement>;
using VectorGrid = Grid<VectorBlock, std::allocator>;
using ScalarGrid = Grid<ScalarBlock, std::allocator>;
using VectorLab = BlockLabOpen<VectorBlock, std::allocator>;
using ScalarLab = BlockLabOpen<ScalarBlock, std::allocator>;

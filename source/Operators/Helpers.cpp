//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Helpers.h"

using namespace cubism;

void IC::operator()(const double dt)
{
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  //const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRHSInfo  = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo  = sim.invRho->getBlocksInfo();

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;  VEL.clear();
    VectorBlock& UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock; UDEF.clear();
    ScalarBlock& CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;  CHI.clear();
    ScalarBlock& PRES= *(ScalarBlock*) presInfo[i].ptrBlock; PRES.clear();
    //VectorBlock&    F= *(VectorBlock*)forceInfo[i].ptrBlock;    F.clear();

    ScalarBlock& TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;  TMP.clear();
    ScalarBlock& PRHS= *(ScalarBlock*) pRHSInfo[i].ptrBlock; PRHS.clear();
    VectorBlock& TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock; TMPV.clear();
    VectorBlock& IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock; IRHO.set(1);
    assert(velInfo[i].blockID ==  uDefInfo[i].blockID);
    assert(velInfo[i].blockID ==   chiInfo[i].blockID);
    assert(velInfo[i].blockID ==  presInfo[i].blockID);
    //assert(velInfo[i].blockID == forceInfo[i].blockID);
    assert(velInfo[i].blockID ==   tmpInfo[i].blockID);
    assert(velInfo[i].blockID ==  pRHSInfo[i].blockID);
    assert(velInfo[i].blockID ==  tmpVInfo[i].blockID);
  }
}

Real findMaxU::run() const
{
  const Real UINF = sim.uinfx, VINF = sim.uinfy;
  Real U = 0, V = 0, u = 0, v = 0;
  #pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  for (size_t i=0; i < Nblocks; i++)
  {
    const VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      U = std::max( U, std::fabs( VEL(ix,iy).u[0] + UINF ) );
      V = std::max( V, std::fabs( VEL(ix,iy).u[1] + VINF ) );
      u = std::max( u, std::fabs( VEL(ix,iy).u[0] ) );
      v = std::max( v, std::fabs( VEL(ix,iy).u[1] ) );
    }
  }
  return std::max( { U, V, u, v } );
}

void Checker::run(std::string when) const
{
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  //const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRHSInfo  = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    VectorBlock& UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock;
    ScalarBlock& CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
    ScalarBlock& PRES= *(ScalarBlock*) presInfo[i].ptrBlock;
    //VectorBlock&    F= *(VectorBlock*)forceInfo[i].ptrBlock;

    ScalarBlock& TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
    ScalarBlock& PRHS= *(ScalarBlock*) pRHSInfo[i].ptrBlock;
    VectorBlock& TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if(std::isnan( VEL(ix,iy).u[0])) {
        printf("isnan( VEL(ix,iy).u[0]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf( VEL(ix,iy).u[0])) {
        printf("isinf( VEL(ix,iy).u[0]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan(UDEF(ix,iy).u[0])) {
        printf("isnan(UDEF(ix,iy).u[0]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf(UDEF(ix,iy).u[0])) {
        printf("isinf(UDEF(ix,iy).u[0]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan(TMPV(ix,iy).u[0])) {
        printf("isnan(TMPV(ix,iy).u[0]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf(TMPV(ix,iy).u[0])) {
        printf("isinf(TMPV(ix,iy).u[0]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan( VEL(ix,iy).u[1])) {
        printf("isnan( VEL(ix,iy).u[1]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf( VEL(ix,iy).u[1])) {
        printf("isinf( VEL(ix,iy).u[1]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan(UDEF(ix,iy).u[1])) {
        printf("isnan(UDEF(ix,iy).u[1]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf(UDEF(ix,iy).u[1])) {
        printf("isinf(UDEF(ix,iy).u[1]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan(TMPV(ix,iy).u[1])) {
        printf("isnan(TMPV(ix,iy).u[1]) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf(TMPV(ix,iy).u[1])) {
        printf("isinf(TMPV(ix,iy).u[1]) %s\n", when.c_str());
        fflush(0); abort();
      }

      if(std::isnan( CHI(ix,iy).s   )) {
        printf("isnan( CHI(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf( CHI(ix,iy).s   )) {
        printf("isinf( CHI(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan(PRES(ix,iy).s   )) {
        printf("isnan(PRES(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf(PRES(ix,iy).s   )) {
        printf("isinf(PRES(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan( TMP(ix,iy).s   )) {
        printf("isnan( TMP(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf( TMP(ix,iy).s   )) {
        printf("isinf( TMP(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isnan(PRHS(ix,iy).s   )) {
        printf("isnan(PRHS(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
      if(std::isinf(PRHS(ix,iy).s   )) {
        printf("isinf(PRHS(ix,iy).s   ) %s\n", when.c_str());
        fflush(0); abort();
      }
    }
  }
}

void ApplyObjVel::operator()(const double dt)
{
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& UF = *(VectorBlock*)  velInfo[i].ptrBlock;
    VectorBlock& US = *(VectorBlock*) uDefInfo[i].ptrBlock;
    ScalarBlock& X  = *(ScalarBlock*)  chiInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
     UF(ix,iy).u[0]= UF(ix,iy).u[0] *(1-X(ix,iy).s) +US(ix,iy).u[0] *X(ix,iy).s;
     UF(ix,iy).u[1]= UF(ix,iy).u[1] *(1-X(ix,iy).s) +US(ix,iy).u[1] *X(ix,iy).s;
    }
  }
}

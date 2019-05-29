//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Helpers.h"
#include <gsl/gsl_linalg.h>
#include <random>

using namespace cubism;


std::array<double, 4> FadeOut::solveMatrix(const double W, const double E,
                                           const double S, const double N) const
{
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  const double NX = sim.bpdx * BSX, NY = sim.bpdy * BSY;
  double A[4][4] = {
      {  NY, 0.0, 1.0, 1.0},
      { 0.0,  NY, 1.0, 1.0},
      { 1.0, 1.0,  NX, 0.0},
      { 1.0, 1.0, 0.0,  NX}
  };

  double b[4] = { W, E, S, N };

  gsl_matrix_view Agsl = gsl_matrix_view_array (&A[0][0], 4, 4);
  gsl_vector_view bgsl = gsl_vector_view_array (b, 4);
  gsl_vector *xgsl = gsl_vector_alloc (4);
  int sgsl;
  gsl_permutation * permgsl = gsl_permutation_alloc (4);
  gsl_linalg_LU_decomp (& Agsl.matrix, permgsl, & sgsl);
  gsl_linalg_LU_solve ( & Agsl.matrix, permgsl, & bgsl.vector, xgsl);

  std::array<double, 4> ret = {gsl_vector_get(xgsl, 0),
                               gsl_vector_get(xgsl, 1),
                               gsl_vector_get(xgsl, 2),
                               gsl_vector_get(xgsl, 3)};
  gsl_permutation_free (permgsl);
  gsl_vector_free (xgsl);
  return ret;
}


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

  #pragma omp parallel
  {
    //std::random_device rd;
    //std::normal_distribution<Real> dist(0, 1e-7);
    //std::mt19937 gen(rd());
    #pragma omp for schedule(static)
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
      ScalarBlock& IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock; IRHO.set(1);
      assert(velInfo[i].blockID ==  uDefInfo[i].blockID);
      assert(velInfo[i].blockID ==   chiInfo[i].blockID);
      assert(velInfo[i].blockID ==  presInfo[i].blockID);
      //assert(velInfo[i].blockID == forceInfo[i].blockID);
      assert(velInfo[i].blockID ==   tmpInfo[i].blockID);
      assert(velInfo[i].blockID ==  pRHSInfo[i].blockID);
      assert(velInfo[i].blockID ==  tmpVInfo[i].blockID);
      //for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      //for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      //  VEL(ix,iy).u[0] += dist(gen);
      //  VEL(ix,iy).u[1] += dist(gen);
      //}
    }
  }
}

void FadeOut::operator()(const double dt)
{
  static constexpr double EPS = std::numeric_limits<Real>::epsilon();
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
  //const auto& extent = sim.extents; const Real H = sim.vel->getH();
  const auto isW = [&](const BlockInfo& info) {
    return info.index[0] == 0;
  };
  const auto isE = [&](const BlockInfo& info) {
    return info.index[0] == sim.bpdx-1;
  };
  const auto isS = [&](const BlockInfo& info) {
    return info.index[1] == 0;
  };
  const auto isN = [&](const BlockInfo& info) {
    return info.index[1] == sim.bpdy-1;
  };
  const Real uinfx = sim.uinfx, uinfy = sim.uinfy;
  const Real normU = std::max( std::sqrt(uinfx*uinfx + uinfy*uinfy), EPS );
  const Real coefW = std::min((Real)1, std::max(normU+uinfx, (Real)0) /normU);
  const Real coefE = std::min((Real)1, std::max(normU-uinfx, (Real)0) /normU);
  const Real coefS = std::min((Real)1, std::max(normU+uinfy, (Real)0) /normU);
  const Real coefN = std::min((Real)1, std::max(normU-uinfy, (Real)0) /normU);

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    if( isW(velInfo[i]) ) // west
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        VEL(BX, iy).u[0] -= coefW * VEL(BX, iy).u[0];
        VEL(BX, iy).u[1] -= coefW * VEL(BX, iy).u[1];
      }
    if( isE(velInfo[i]) ) // east
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        VEL(EX, iy).u[0] -= coefE * VEL(EX, iy).u[0];
        VEL(EX, iy).u[1] -= coefE * VEL(EX, iy).u[1];
      }
    if( isS(velInfo[i]) ) // south
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        VEL(ix, BY).u[0] -= coefS * VEL(ix, BY).u[0];
        VEL(ix, BY).u[1] -= coefS * VEL(ix, BY).u[1];
      }
    if( isN(velInfo[i]) ) // north
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        VEL(ix, EY).u[0] -= coefN * VEL(ix, EY).u[0];
        VEL(ix, EY).u[1] -= coefN * VEL(ix, EY).u[1];
      }
  }
}

Real findMaxU::run() const
{
  const Real UINF = sim.uinfx, VINF = sim.uinfy;
  ///*
  #ifdef ZERO_TOTAL_MOM
  const std::vector<BlockInfo>& iRhoInfo  = sim.invRho->getBlocksInfo();
  Real momX = 0, momY = 0, totM = 0; const Real h = sim.getH();
  #pragma omp parallel for schedule(static) reduction(+ : momX, momY, totM)
  for (size_t i=0; i < Nblocks; i++) {
    const ScalarBlock& IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;
    const VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      const Real facMom = h*h / IRHO(ix,iy).s;
      momX += facMom * VEL(ix,iy).u[0];
      momY += facMom * VEL(ix,iy).u[1];
      totM += facMom;
    }
  }
  //printf("Integral of momenta X:%e Y:%e mass:%e\n", momX, momY, totM);
  const Real DU = momX / totM, DV = momY / totM;
  #endif
  //*/
  Real U = 0, V = 0, u = 0, v = 0;
  #pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      #ifdef ZERO_TOTAL_MOM
        VEL(ix,iy).u[0] -= DU; VEL(ix,iy).u[1] -= DV;
      #endif
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

//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Helpers.h"
#include <gsl/gsl_linalg.h>

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
    ScalarBlock& IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock; IRHO.set(1);
    assert(velInfo[i].blockID ==  uDefInfo[i].blockID);
    assert(velInfo[i].blockID ==   chiInfo[i].blockID);
    assert(velInfo[i].blockID ==  presInfo[i].blockID);
    //assert(velInfo[i].blockID == forceInfo[i].blockID);
    assert(velInfo[i].blockID ==   tmpInfo[i].blockID);
    assert(velInfo[i].blockID ==  pRHSInfo[i].blockID);
    assert(velInfo[i].blockID ==  tmpVInfo[i].blockID);
  }
}

void FadeOut::operator()(const double dt)
{
  //static constexpr double EPS = std::numeric_limits<Real>::epsilon();
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr int endX = BSX-1, endY = BSY-1;
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

  {
  Real Uw=0, Vw=0, Ue=0, Ve=0, Us=0, Vs=0, Un=0, Vn=0;
  #pragma omp parallel for schedule(dynamic) reduction(+:Uw,Vw,Ue,Ve,Us,Vs,Un,Vn)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    if( isW(velInfo[i]) ) // west
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        if(sim.uinfx>0) VEL(   0,   iy).u[0] = 0;
        Uw += VEL(   0,   iy).u[0];
        Vw += VEL(   0,   iy).u[1]; }
    if( isE(velInfo[i]) ) // east
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        if(sim.uinfx<0) VEL(endX,   iy).u[0] = 0;
        Ue += VEL(endX,   iy).u[0];
        Ve += VEL(endX,   iy).u[1]; }
    if( isS(velInfo[i]) ) // south
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        if(sim.uinfy>0) VEL(  ix,    0).u[1] = 0;
        Us += VEL(  ix,    0).u[0];
        Vs += VEL(  ix,    0).u[1]; }
    if( isN(velInfo[i]) ) // north
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        if(sim.uinfy<0) VEL(  ix, endY).u[1] = 0;
        Un += VEL(  ix, endY).u[0];
        Vn += VEL(  ix, endY).u[1]; }
  }
  //printf("correction w:[%e %e] e:[%e %e] s:[%e %e] n:[%e %e]\n", Uw,Vw,Ue,Ve,Us,Vs,Un,Vn);

  #if 1
  const auto NX = sim.bpdx * BSX, NY = sim.bpdy * BSY;
  const Real corrUw = (NX*NY*Uw - 2*Uw + 2*Ue - NY*Un - NY*Us)/(NY*(NX*NY - 4));
  const Real corrUe = (NX*NY*Ue - 2*Ue + 2*Uw - NY*Un - NY*Us)/(NY*(NX*NY - 4));
  const Real corrUs = (NX*NY*Us - 2*Us + 2*Un - NX*Ue - NX*Uw)/(NX*(NX*NY - 4));
  const Real corrUn = (NX*NY*Un - 2*Un + 2*Us - NX*Ue - NX*Uw)/(NX*(NX*NY - 4));
  const Real corrVw = (NX*NY*Vw - 2*Vw + 2*Ve - NY*Vn - NY*Vs)/(NY*(NX*NY - 4));
  const Real corrVe = (NX*NY*Ve - 2*Ve + 2*Vw - NY*Vn - NY*Vs)/(NY*(NX*NY - 4));
  const Real corrVs = (NX*NY*Vs - 2*Vs + 2*Vn - NX*Ve - NX*Vw)/(NX*(NX*NY - 4));
  const Real corrVn = (NX*NY*Vn - 2*Vn + 2*Vs - NX*Ve - NX*Vw)/(NX*(NX*NY - 4));
  #else
  const auto CU = solveMatrix(Uw, Ue, Us, Un);
  const auto CV = solveMatrix(Vw, Ve, Vs, Vn);
  const Real corrUw = CU[0], corrUe = CU[1], corrUs = CU[2], corrUn = CU[3];
  const Real corrVw = CV[0], corrVe = CV[1], corrVs = CV[2], corrVn = CV[3];
  #endif

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    if( isW(velInfo[i]) ) // west
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        VEL(   0,   iy).u[0] -= corrUw; VEL(   0,   iy).u[1] -= corrVw; }
    if( isE(velInfo[i]) ) // east
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        VEL(endX,   iy).u[0] -= corrUe; VEL(endX,   iy).u[1] -= corrVe; }
    if( isS(velInfo[i]) ) // south
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        VEL(  ix,    0).u[0] -= corrUs; VEL(  ix,    0).u[1] -= corrVs; }
    if( isN(velInfo[i]) ) // north
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        VEL(  ix, endY).u[0] -= corrUn; VEL(  ix, endY).u[1] -= corrVn; }
  }
  }
  if(0)
  {
    Real Uw=0, Vw=0, Ue=0, Ve=0, Us=0, Vs=0, Un=0, Vn=0;
    #pragma omp parallel for schedule(dynamic) reduction(+:Uw,Vw,Ue,Ve,Us,Vs,Un,Vn)
    for (size_t i=0; i < Nblocks; i++)
    {
      const VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
      if( isW(velInfo[i]) ) // west
        for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
          Uw += VEL(   0,   iy).u[0]; Vw += VEL(   0,   iy).u[1]; }
      if( isE(velInfo[i]) ) // east
        for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
          Ue += VEL(endX,   iy).u[0]; Ve += VEL(endX,   iy).u[1]; }
      if( isS(velInfo[i]) ) // south
        for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
          Us += VEL(  ix,    0).u[0]; Vs += VEL(  ix,    0).u[1]; }
      if( isN(velInfo[i]) ) // north
        for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
          Un += VEL(  ix, endY).u[0]; Vn += VEL(  ix, endY).u[1]; }
    }
    printf("after correction w:[%e %e] e:[%e %e] s:[%e %e] n:[%e %e]\n", Uw,Vw,Ue,Ve,Us,Vs,Un,Vn);
  }
  /*
  const Real fadeLenX = sim.fadeLenX, fadeLenY = sim.fadeLenY;
  const Real invFadeX = 1/(fadeLenX+EPS), invFadeY = 1/(fadeLenY+EPS);
  const auto _is_touching = [&] (const BlockInfo& i)
  {
    Real min_pos[2], max_pos[2];
    i.pos(max_pos, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    i.pos(min_pos, 0, 0);
    const bool touchW = fadeLenX >= min_pos[0];
    const bool touchE = fadeLenX >= extent[0] - max_pos[0];
    const bool touchS = fadeLenY >= min_pos[1];
    const bool touchN = fadeLenY >= extent[1] - max_pos[1];
    return touchN || touchE || touchS || touchW;
  };

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    if( not _is_touching(velInfo[i]) ) continue;
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Real p[2]; velInfo[i].pos(p, ix, iy);
      const Real yt = invFadeY*std::max(Real(0), fadeLenY - extent[1] + p[1] );
      const Real yb = invFadeY*std::max(Real(0), fadeLenY - p[1] );
      const Real xt = invFadeX*std::max(Real(0), fadeLenX - extent[0] + p[0] );
      const Real xb = invFadeX*std::max(Real(0), fadeLenX - p[0] );
      const Real killWidth = std::min( std::max({yt, yb, xt, xb}), (Real) 1);
      const Real killFactor = 1 - std::pow(killWidth, 2);
      VEL(ix,iy).u[0] *= killFactor;
      VEL(ix,iy).u[1] *= killFactor;
    }
  }
  */
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
